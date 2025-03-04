import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Node:
    """Represents a node (joint) in the structure."""
    name: str
    x: float
    y: float
    is_support: bool = False
    support_type: str = None  # 'fixed', 'pinned', 'roller'

@dataclass
class Member:
    """Represents a member in the structure."""
    name: str
    start_node: Node
    end_node: Node
    EI: float  # Flexural rigidity (E*I)
    length: float = None
    
    def __post_init__(self):
        if self.length is None:
            dx = self.end_node.x - self.start_node.x
            dy = self.end_node.y - self.start_node.y
            self.length = np.sqrt(dx**2 + dy**2)

@dataclass
class Load:
    """Represents a load on a member."""
    member: Member
    load_type: str  # 'point', 'uniform', 'moment'
    magnitude: float
    location: float = None  # For point loads and moments, distance from start node
    
class StructureAnalyzer:

    """
        Initialize the analyzer with a specific sway case.
        
        Sway Cases:
        1: Basic sway with horizontal loading
        2: Sway with distributed loads at top
        3: Sway with axial load P at top
        4: No sway, axial load at top
        5: Sway with axial load and distributed load at top
        """
    
    def __init__(self, sway_case=1):
        self.nodes = {}  # Changed from list to dict
        self.members = {}  # Changed from list to dict
        self.loads = []
        self.unknowns = {}
        self.equations = []
        self.results = {}
        self.sway_case = sway_case
    
    def add_node(self, name: str, x: float, y: float, is_support: bool = False, support_type: str = None):
        """Add a node to the structure."""
        node = Node(name, x, y, is_support, support_type)
        self.nodes[name] = node  # Store by name instead of appending to list
        return node
    
    def add_member(self, name: str, start_node: Node, end_node: Node, EI: float):
        """Add a member to the structure."""
        member = Member(name, start_node, end_node, EI)
        self.members[name] = member  # Store by name instead of appending to list
        return member
    
    def add_load(self, member: Member, load_type: str, magnitude: float, location: float = None):
        """Add a load to a member."""
        # Auto-set location to midspan for 'point' loads if not provided
        if load_type == 'point' and location is None:
            location = member.length / 2  # Default to center
        load = Load(member, load_type, magnitude, location)
        self.loads.append(load)
        return load
    
    def calculate_fixed_end_moments(self, member: Member) -> Tuple[float, float]:
        """Calculate the fixed end moments for a member due to applied loads."""
        MFAB = 0  # Fixed end moment at start node
        MFBA = 0  # Fixed end moment at end node
        
        # Find all loads on this member
        member_loads = [load for load in self.loads if load.member.name == member.name]
        
        for load in member_loads:
            L = member.length
            
            if load.load_type == 'uniform':
                w = load.magnitude
                # Fixed end moments for uniformly distributed load
                MFAB -= (w * L**2) / 12
                MFBA += (w * L**2) / 12

            elif load.load_type == 'point':
                P = load.magnitude
                
                # Fixed end moments for point load
                MFAB -= (P * L) / 8
                MFBA += (P * L) / 8
                
            elif load.load_type == 'point_x':
                P = load.magnitude
                a = load.location
                b = L - a
                
                # Fixed end moments for point load at a distance
                MFAB -= (P * a * b**2) / L**2
                MFBA += (P * a**2 * b) / L**2
                
            elif load.load_type == 'double_point':
                P = load.magnitude
                
                # Fixed end moments for point load
                MFAB -= (2 * P * L) / 9
                MFBA += (2 * P * L) / 9

            elif load.load_type == 'half_uniform':
                w = load.magnitude
                # Fixed end moments for uniformly distributed load
                MFAB -= (11 * w * L**2) / 192
                MFBA += (5 * w * L**2) / 192

            elif load.load_type == 'moment':
                M = load.magnitude
                a = load.location
                b = L - a
                
                # Fixed end moments for applied moment
                MFAB -= (M * b) / L
                MFBA -= (M * a) / L
        
        return MFAB, MFBA
    
    def setup_equations(self):
        """Set up the equilibrium equations for the structure."""
        self.unknowns = {}
        self.equations = []

        # ==== Adjustment 1: Correct Free End Detection ====
        # Free ends are nodes with only ONE connected member and not supports
        free_ends = []
        for node in self.nodes.values():
            connected = self.get_connected_members(node)
            if len(connected) == 1 and not node.is_support:
                free_ends.append(node.name)

        # ==== Adjustment 2: Smarter Sway Detection ====
        # Check if horizontal translation is possible (true sway frames only)
        vertical_members = [m for m in self.members.values() 
                          if m.start_node.y != m.end_node.y]
        horizontal_restrained = any(
            n.support_type in ['fixed', 'pinned', 'roller'] 
            for n in self.nodes.values()
        )
        has_sidesway = (len(vertical_members) > 0 and not horizontal_restrained)

        # Add unknown rotations for non-fixed, non-free nodes
        for node in self.nodes.values():
            if node.support_type != 'fixed' and node.name not in free_ends:
                self.unknowns[f"theta_{node.name}"] = 0

        # Add sway unknown only if truly needed
        if has_sidesway:
            self.unknowns["delta"] = 0
            
        # Create equilibrium equations only for non-free nodes
        for node in self.nodes.values():
            # Skip free ends entirely
            if node.name in free_ends:
                continue

            # Process nodes that are not free ends
            if not node.is_support or node.support_type != 'fixed':
                equation = {'constant': 0}
                connected_members = self.get_connected_members(node)

                for member in connected_members:
                    L = member.length
                    EI = member.EI
                    k = EI / L  # Stiffness coefficient

                    # Determine if node is start or end of the member
                    is_start = (member.start_node == node)
                    other_node = member.end_node if is_start else member.start_node

                    # Skip contributions from free-end nodes
                    if other_node.name in free_ends:
                        continue

                    # Fixed end moments
                    MFAB, MFBA = self.calculate_fixed_end_moments(member)

                    # Add fixed end moment to equation constant
                    if is_start:
                        equation['constant'] += MFAB
                    else:
                        equation['constant'] += MFBA

                    # Slope-deflection terms (only include non-free nodes)
                    current_theta = f"theta_{node.name}"
                    other_theta = f"theta_{other_node.name}"
                    
                    equation[current_theta] = equation.get(current_theta, 0) + 4 * k
                    if other_theta in self.unknowns:
                        equation[other_theta] = equation.get(other_theta, 0) + 2 * k

                    # Sway terms (if applicable)
                    if "delta" in self.unknowns and member.start_node.y != member.end_node.y:
                        sway_term = -6 * k / L
                        equation["delta"] = equation.get("delta", 0) + sway_term

                if equation != {'constant': 0}:
                    self.equations.append(equation)

        # Add global sway equilibrium equation if needed
        if has_sidesway and "delta" in self.unknowns:
            sway_equation = {'constant': 0}
            for member in self.members.values():
                if member.start_node.y == member.end_node.y:
                    continue  # Skip horizontal members

                L = member.length
                EI = member.EI
                k = 6 * EI / L**2

                # Only include non-free nodes
                start_name = member.start_node.name
                end_name = member.end_node.name
                if start_name not in free_ends:
                    sway_equation[f"theta_{start_name}"] = sway_equation.get(f"theta_{start_name}", 0) + k
                if end_name not in free_ends:
                    sway_equation[f"theta_{end_name}"] = sway_equation.get(f"theta_{end_name}", 0) + k
                sway_equation["delta"] = sway_equation.get("delta", 0) - 12 * EI / L**3

            self.equations.append(sway_equation)
    
    def get_connected_members(self, node: Node) -> List[Member]:
        """Get all members connected to a node."""
        return [m for m in self.members.values() if m.start_node == node or m.end_node == node]
    
    def solve_equations(self):
        """Solve the system of equations to find unknown rotations and displacements."""
        n = len(self.unknowns)
        num_eq = len(self.equations)
        
        if num_eq != n:
            print(f"Error: {num_eq} equations but {n} unknowns. System is unbalanced.")
            return False
        
        A = np.zeros((n, n))
        b = np.zeros(n)
        unknown_keys = list(self.unknowns.keys())
        
        for i, eq in enumerate(self.equations):
            if i >= n:
                print(f"Error: Equation index {i} exceeds unknowns count {n}.")
                return False
            b[i] = -eq.get("constant", 0)
            for j, key in enumerate(unknown_keys):
                A[i, j] = eq.get(key, 0)
        
        try:
            x = np.linalg.solve(A, b)
            for i, key in enumerate(unknown_keys):
                self.unknowns[key] = x[i]
            return True
        except np.linalg.LinAlgError:
            print("Error: Singular matrix. Check for unstable structure or redundant constraints.")
            return False
    
    def calculate_member_end_moments(self):
        """Calculate the final end moments for all members."""
        for member in self.members.values():
            # Get fixed end moments
            MFAB, MFBA = self.calculate_fixed_end_moments(member)
            
            # Get rotations
            theta_A = self.unknowns.get(f"theta_{member.start_node.name}", 0)
            theta_B = self.unknowns.get(f"theta_{member.end_node.name}", 0)
            
            # Get sidesway if applicable
            delta = self.unknowns.get("delta", 0)
            
            # Calculate end moments using slope deflection equations
            k = member.EI / member.length
            
            # Check if we need to include sidesway term
            sidesway_term = 0
            if "delta" in self.unknowns and member.start_node.y != member.end_node.y:
                sidesway_term = -3 * delta / member.length
            
            # Calculate end moments
            MAB = MFAB + 2 * k * (2 * theta_A + theta_B + sidesway_term)
            MBA = MFBA + 2 * k * (theta_A + 2 * theta_B + sidesway_term)
            
            # Store results
            self.results[f"M_{member.name}_start"] = MAB
            self.results[f"M_{member.name}_end"] = MBA
    
    def calculate_shear_forces(self):
        """Calculate shear forces for all members."""
        for member in self.members.values():
            # Get end moments
            MAB = self.results[f"M_{member.name}_start"]
            MBA = self.results[f"M_{member.name}_end"]
            
            # Initialize shear forces at both ends
            VAB = 0
            VBA = 0
            
            # Find all loads on this member
            member_loads = [load for load in self.loads if load.member == member]
            
            # Calculate reactions due to loads
            for load in member_loads:
                L = member.length
                
                if load.load_type == 'uniform':
                    w = load.magnitude
                    # Uniform load contributes to shear at both ends
                    VAB += w * L / 2
                    VBA += w * L / 2
                    
                elif load.load_type == 'point':
                    P = load.magnitude
                    a = load.location
                    b = L - a
                    
                    # Point load contributions
                    VAB += P * b / L
                    VBA += P * a / L
            
            # Add contribution from end moments
            VAB += (MAB + MBA) / member.length
            VBA -= (MAB + MBA) / member.length
            
            # Store results
            self.results[f"V_{member.name}_start"] = VAB
            self.results[f"V_{member.name}_end"] = VBA
    
    def analyze(self):
        """Run the complete analysis process."""
        self.setup_equations()
        if self.solve_equations():
            self.calculate_member_end_moments()
            self.calculate_shear_forces()
            return True
        return False
    
    def plot_results(self):
        """Plot the structure with bending moment and shear force diagrams."""
        # Plot structure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Structure
        plt.subplot(3, 1, 1)
        plt.title('Structure')
        
        # Plot nodes
        for node in self.nodes.values():
            plt.plot(node.x, node.y, 'o', markersize=10)
            plt.text(node.x, node.y + 0.2, node.name)
        
        # Plot members
        for member in self.members.values():
            plt.plot([member.start_node.x, member.end_node.x], 
                     [member.start_node.y, member.end_node.y], 'k-', linewidth=2)
            
            # Plot loads
            member_loads = [load for load in self.loads if load.member == member]
            for load in member_loads:
                if load.load_type == 'uniform':
                    # Draw uniform load
                    x_start, y_start = member.start_node.x, member.start_node.y
                    x_end, y_end = member.end_node.x, member.end_node.y
                    
                    # Draw load arrows
                    arrow_spacing = member.length / 10
                    num_arrows = int(member.length / arrow_spacing)
                    
                    for i in range(num_arrows + 1):
                        t = i / num_arrows
                        x = x_start + t * (x_end - x_start)
                        y = y_start + t * (y_end - y_start)
                        
                        # Draw arrow (simplified for horizontal members)
                        arrow_length = 0.3
                        if abs(y_end - y_start) < 0.001:  # Horizontal member
                            plt.arrow(x, y, 0, -arrow_length, head_width=0.1, head_length=0.05, 
                                      fc='r', ec='r', length_includes_head=True)
                        else:  # Vertical member
                            plt.arrow(x, y, -arrow_length, 0, head_width=0.1, head_length=0.05, 
                                      fc='r', ec='r', length_includes_head=True)
                
                elif load.load_type == 'point':
                    # Draw point load
                    t = load.location / member.length
                    x = member.start_node.x + t * (member.end_node.x - member.start_node.x)
                    y = member.start_node.y + t * (member.end_node.y - member.start_node.y)
                    
                    # Draw arrow
                    arrow_length = 0.5
                    plt.arrow(x, y, 0, -arrow_length, head_width=0.15, head_length=0.1, 
                              fc='b', ec='b', length_includes_head=True)
        
        plt.grid(True)
        plt.axis('equal')
        
        # Plot 2: Bending Moment Diagram
        plt.subplot(3, 1, 2)
        plt.title('Bending Moment Diagram')
        
        for member in self.members.values():
            # Get end moments
            MAB = self.results[f"M_{member.name}_start"]
            MBA = self.results[f"M_{member.name}_end"]
            
            # Create x coordinates along the member
            num_points = 100
            x_coords = np.linspace(0, member.length, num_points)
            
            # Calculate moment at each point
            moments = np.zeros(num_points)
            
            # End moments contribution
            moments += MAB * (1 - x_coords / member.length) + MBA * (x_coords / member.length)
            
            # Loads contribution
            member_loads = [load for load in self.loads if load.member == member]
            for load in member_loads:
                if load.load_type == 'uniform':
                    w = load.magnitude
                    moments += w * x_coords * (member.length - x_coords) / 2
                
                elif load.load_type == 'point':
                    P = load.magnitude
                    a = load.location
                    moments += np.where(x_coords <= a, 
                                        P * x_coords * (1 - a / member.length), 
                                        P * a * (1 - x_coords / member.length))
            
            # Calculate global coordinates
            x_global = member.start_node.x + (member.end_node.x - member.start_node.x) * x_coords / member.length
            y_global = member.start_node.y + (member.end_node.y - member.start_node.y) * x_coords / member.length
            
            # Plot the moment diagram
            plt.plot(x_global, y_global + moments/50, 'r-')  # Scale factor for visibility
            
            # Fill between the moment curve and the member axis
            plt.fill_between(x_global, y_global, y_global + moments/50, alpha=0.3)
        
        plt.grid(True)
        
        # Plot 3: Shear Force Diagram
        plt.subplot(3, 1, 3)
        plt.title('Shear Force Diagram')
        
        for member in self.members.values():
            # Get end shears
            VAB = self.results[f"V_{member.name}_start"]
            VBA = self.results[f"V_{member.name}_end"]
            
            # Create x coordinates along the member
            num_points = 100
            x_coords = np.linspace(0, member.length, num_points)
            
            # Calculate shear at each point
            shears = np.zeros(num_points)
            
            # End shears contribution (simplified linear variation for demonstration)
            shears += VAB * (1 - x_coords / member.length) + VBA * (x_coords / member.length)
            
            # Calculate global coordinates
            x_global = member.start_node.x + (member.end_node.x - member.start_node.x) * x_coords / member.length
            y_global = member.start_node.y + (member.end_node.y - member.start_node.y) * x_coords / member.length
            
            # Plot the shear diagram
            plt.plot(x_global, y_global + shears/50, 'b-')  # Scale factor for visibility
            
            # Fill between the shear curve and the member axis
            plt.fill_between(x_global, y_global, y_global + shears/50, alpha=0.3)
        
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def print_results(self):
        """Print analysis results."""
        print("\n==== Analysis Results ====")
        
        # Print unknown rotations and displacements
        print("\nUnknown Rotations and Displacements:")
        for key, value in self.unknowns.items():
            print(f"{key} = {value:.6f}")
        
        # Print member end moments
        print("\nMember End Moments:")
        for member in self.members.values():
            MAB = self.results[f"M_{member.name}_start"]
            MBA = self.results[f"M_{member.name}_end"]
            print(f"Member {member.name}:")
            print(f"  M_{member.start_node.name}{member.end_node.name} = {MAB:.2f} kNm")
            print(f"  M_{member.end_node.name}{member.start_node.name} = {MBA:.2f} kNm")
        
        # Print member shear forces
        print("\nMember Shear Forces:")
        for member in self.members.values():
            VAB = self.results[f"V_{member.name}_start"]
            VBA = self.results[f"V_{member.name}_end"]
            print(f"Member {member.name}:")
            print(f"  V_{member.start_node.name}{member.end_node.name} = {VAB:.2f} kN")
            print(f"  V_{member.end_node.name}{member.start_node.name} = {VBA:.2f} kN")


# Example usage - Simple continuous beam
def example_continuous_beam():
    analyzer = StructureAnalyzer()
    
    # Define nodes
    node_A = analyzer.add_node("A", 0, 0, is_support=True, support_type="fixed")
    node_B = analyzer.add_node("B", 5, 0)
    node_C = analyzer.add_node("C", 10, 0)
    node_D = analyzer.add_node("D", 15, 0, is_support=True, support_type="fixed")
    
    # Define members
    EI = 1000  # Assumed constant EI value
    member_AB = analyzer.add_member("AB", node_A, node_B, EI)
    member_BC = analyzer.add_member("BC", node_B, node_C, 2*EI)  # 2I for the middle section
    member_CD = analyzer.add_member("CD", node_C, node_D, EI)
    
    # Add loads
    analyzer.add_load(member_AB, "uniform", 20)  # 20 kN/m
    analyzer.add_load(member_BC, "point", 80, 2)  # 80 kN at 2m from B
    analyzer.add_load(member_CD, "uniform", 15)  # 15 kN/m
    
    # Run analysis
    if analyzer.analyze():
        analyzer.print_results()
        analyzer.plot_results()
    else:
        print("Analysis failed.")

# Example usage - Portal frame
def example_portal_frame():
    analyzer = StructureAnalyzer()
    
    # Define nodes
    node_A = analyzer.add_node("A", 0, 0, is_support=True, support_type="fixed")
    node_B = analyzer.add_node("B", 0, 5, is_support=False)
    node_C = analyzer.add_node("C", 4, 5, is_support=False)
    node_D = analyzer.add_node("D", 4, 0, is_support=True, support_type="fixed")
    
    # Define members
    EI = 1000  # Assumed constant EI value
    member_AB = analyzer.add_member("AB", node_A, node_B, EI)
    member_BC = analyzer.add_member("BC", node_B, node_C, EI)
    member_CD = analyzer.add_member("CD", node_C, node_D, EI)
    
    # Add loads
    analyzer.add_load(member_BC, "point", -50, 2)  # 50 kN at mid-span
    
    # Run analysis
    if analyzer.analyze():
        analyzer.print_results()
        analyzer.plot_results()
    else:
        print("Analysis failed.")

# Run examples
if __name__ == "__main__":
    print("Example 1: Continuous Beam Analysis")
    example_continuous_beam()
    
    print("\nExample 2: Portal Frame Analysis")
    example_portal_frame()