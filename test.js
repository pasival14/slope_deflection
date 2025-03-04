// Global variables
let sessionId = null;
let nodes = {};
let members = {};
let loads = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Create a new session
    createSession();
    
    // Setup event listeners
    document.getElementById('nodeForm').addEventListener('submit', addNode);
    document.getElementById('memberForm').addEventListener('submit', addMember);
    document.getElementById('loadForm').addEventListener('submit', addLoad);
    document.getElementById('isSupport').addEventListener('change', toggleSupportType);
    document.getElementById('loadType').addEventListener('change', toggleLoadLocation);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeStructure);
    document.getElementById('clearBtn').addEventListener('click', clearStructure);
    document.getElementById('exampleSelect').addEventListener('change', loadExample);
});

// Create a new session
function createSession() {
    // For local testing without a server, we can create a mock session
    sessionId = 'local-' + Date.now();
    console.log('Mock session created with ID:', sessionId);
    
    // In a real application, this would call the server:
    /*
    fetch('/api/create_session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        sessionId = data.session_id;
        console.log('Session created with ID:', sessionId);
    })
    .catch(error => {
        console.error('Error creating session:', error);
        showAlert('error', 'Failed to initialize application. Please refresh the page.');
    });
    */
}

// Toggle support type dropdown based on checkbox
function toggleSupportType() {
    const isSupport = document.getElementById('isSupport').checked;
    const supportType = document.getElementById('supportType');
    
    supportType.disabled = !isSupport;
    
    if (!isSupport) {
        supportType.value = '';
    }
}

document.getElementById('isSupport').onchange = function() {
    const supportType = document.getElementById('supportType');
    supportType.disabled = !this.checked;
};

// Toggle load location field based on load type
function toggleLoadLocation() {
    const loadType = document.getElementById('loadType').value;
    const locationField = document.getElementById('locationField');
    
    // Only point and moment loads need a location
    if (loadType === 'point' || loadType === 'moment') {
        locationField.style.display = 'block';
        document.getElementById('loadLocation').required = true;
    } else {
        locationField.style.display = 'none';
        document.getElementById('loadLocation').required = false;
        document.getElementById('loadLocation').value = '';
    }
}

// Add a node to the structure
function addNode(event) {
    event.preventDefault();
    
    const name = document.getElementById('nodeName').value;
    const x = parseFloat(document.getElementById('nodeX').value);
    const y = parseFloat(document.getElementById('nodeY').value);
    const isSupport = document.getElementById('isSupport').checked;
    const supportType = isSupport ? document.getElementById('supportType').value : null;

    // Store node data
    nodes[name] = { x, y, isSupport, supportType };
    
    console.log('Node added:', nodes[name]);
    
    // Update the nodes list display
    updateNodesList();
    
    // Update the node selections in member form
    updateNodeSelections();
    
    showAlert('success', 'Node added successfully.');
    document.getElementById('nodeForm').reset();
    document.getElementById('supportType').disabled = true;
}

// Add a member to the structure
function addMember(event) {
    event.preventDefault();
    
    const name = document.getElementById('memberName').value;
    const startNode = document.getElementById('startNode').value;
    const endNode = document.getElementById('endNode').value;
    const EI = parseFloat(document.getElementById('EI').value);
    
    // Validate inputs
    if (startNode === endNode) {
        showAlert('error', 'Start and end nodes must be different.');
        return;
    }
    
    // Store member data
    members[name] = { 
        startNode, 
        endNode, 
        EI,
        // Calculate length for display
        length: calculateLength(nodes[startNode], nodes[endNode])
    };
    
    console.log('Member added:', members[name]);
    
    // Update the members list display
    updateMembersList();
    
    // Update the member selections in load form
    updateMemberSelections();
    
    showAlert('success', 'Member added successfully.');
    document.getElementById('memberForm').reset();
}

// Add a load to the structure
function addLoad(event) {
    event.preventDefault();
    
    const memberName = document.getElementById('loadMember').value;
    const loadType = document.getElementById('loadType').value;
    const magnitude = parseFloat(document.getElementById('loadMagnitude').value);
    let location = null;
    
    if (loadType === 'point' || loadType === 'moment') {
        location = parseFloat(document.getElementById('loadLocation').value);
        
        // Validate location
        if (isNaN(location)) {
            showAlert('error', 'Location is required for point and moment loads.');
            return;
        }
        
        // Check if location is within member length
        const memberLength = members[memberName].length;
        if (location < 0 || location > memberLength) {
            showAlert('error', `Location must be between 0 and ${memberLength} meters.`);
            return;
        }
    }
    
    // Generate a unique ID for the load
    const loadId = 'load-' + Date.now();
    
    // Store load data
    loads[loadId] = { memberName, loadType, magnitude, location };
    
    console.log('Load added:', loads[loadId]);
    
    // Update the loads list display
    updateLoadsList();
    
    showAlert('success', 'Load added successfully.');
    document.getElementById('loadForm').reset();
    
    // Reset the location field visibility
    toggleLoadLocation();
}

// Analyze the structure
function analyzeStructure() {
    // Check if we have enough data to analyze
    if (Object.keys(nodes).length < 2) {
        showAlert('error', 'At least two nodes are required for analysis.');
        return;
    }
    
    if (Object.keys(members).length < 1) {
        showAlert('error', 'At least one member is required for analysis.');
        return;
    }
    
    // In local mode, just show a message
    showAlert('info', 'Analysis would be performed here. In a full application, this would send data to the server.');
    
    // Display a placeholder for the visualization
    document.getElementById('visualization').src = 'https://via.placeholder.com/800x600.png?text=Analysis+Results+Would+Appear+Here';
    
    // Display mock results
    displayMockResults();
}

// Clear the structure
function clearStructure() {
    // Clear data
    nodes = {};
    members = {};
    loads = {};
    
    // Clear UI
    document.getElementById('nodesList').innerHTML = '';
    document.getElementById('membersList').innerHTML = '';
    document.getElementById('loadsList').innerHTML = '';
    document.getElementById('visualization').src = '';
    document.getElementById('resultsContainer').innerHTML = 
        '<div class="alert alert-info">Define your structure and click "Analyze Structure" to see results.</div>';
    
    // Reset forms
    document.getElementById('nodeForm').reset();
    document.getElementById('memberForm').reset();
    document.getElementById('loadForm').reset();
    document.getElementById('supportType').disabled = true;
    
    // Clear dropdowns
    updateNodeSelections();
    updateMemberSelections();
    
    showAlert('info', 'Structure cleared.');
}

// Load an example structure
function loadExample() {
    const exampleName = document.getElementById('exampleSelect').value;
    
    if (!exampleName) return;
    
    // Clear current structure
    clearStructure();
    
    if (exampleName === 'continuous_beam') {
        loadContinuousBeamExample();
    } else if (exampleName === 'portal_frame') {
        loadPortalFrameExample();
    }
}

// Helper functions

// Update the nodes list display
function updateNodesList() {
    const nodesList = document.getElementById('nodesList');
    nodesList.innerHTML = '';
    
    for (const [name, node] of Object.entries(nodes)) {
        const nodeItem = document.createElement('li');
        nodeItem.className = 'list-group-item';
        nodeItem.textContent = `${name} (${node.x}, ${node.y}) ${node.isSupport ? 'Support: ' + node.supportType : ''}`;
        nodesList.appendChild(nodeItem);
    }
}

// Update the members list display
function updateMembersList() {
    const membersList = document.getElementById('membersList');
    membersList.innerHTML = '';
    
    for (const [name, member] of Object.entries(members)) {
        const memberItem = document.createElement('li');
        memberItem.className = 'list-group-item';
        memberItem.textContent = `${name}: ${member.startNode} to ${member.endNode}, EI=${member.EI}, L=${member.length.toFixed(2)}m`;
        membersList.appendChild(memberItem);
    }
}

// Update the loads list display
function updateLoadsList() {
    const loadsList = document.getElementById('loadsList');
    loadsList.innerHTML = '';
    
    for (const [id, load] of Object.entries(loads)) {
        const loadItem = document.createElement('li');
        loadItem.className = 'list-group-item';
        
        let loadText = `Member ${load.memberName}: ${load.loadType} load, ${load.magnitude} `;
        
        if (load.loadType === 'uniform') {
            loadText += 'kN/m';
        } else if (load.loadType === 'point') {
            loadText += `kN at ${load.location}m`;
        } else if (load.loadType === 'moment') {
            loadText += `kNm at ${load.location}m`;
        }
        
        loadItem.textContent = loadText;
        loadsList.appendChild(loadItem);
    }
}

// Update node selections in dropdowns
function updateNodeSelections() {
    const startNodeSelect = document.getElementById('startNode');
    const endNodeSelect = document.getElementById('endNode');
    
    // Save current selections
    const startNodeValue = startNodeSelect.value;
    const endNodeValue = endNodeSelect.value;
    
    // Clear options except the first one
    while (startNodeSelect.options.length > 1) {
        startNodeSelect.remove(1);
    }
    
    while (endNodeSelect.options.length > 1) {
        endNodeSelect.remove(1);
    }
    
    // Add new options
    for (const name of Object.keys(nodes)) {
        const startOption = document.createElement('option');
        startOption.value = name;
        startOption.textContent = name;
        startNodeSelect.appendChild(startOption);
        
        const endOption = document.createElement('option');
        endOption.value = name;
        endOption.textContent = name;
        endNodeSelect.appendChild(endOption);
    }
    
    // Restore previous selections if possible
    if (startNodeValue && Object.keys(nodes).includes(startNodeValue)) {
        startNodeSelect.value = startNodeValue;
    }
    
    if (endNodeValue && Object.keys(nodes).includes(endNodeValue)) {
        endNodeSelect.value = endNodeValue;
    }
}

// Update member selections in load form
function updateMemberSelections() {
    const memberSelect = document.getElementById('loadMember');
    
    // Save current selection
    const memberValue = memberSelect.value;
    
    // Clear options except the first one
    while (memberSelect.options.length > 1) {
        memberSelect.remove(1);
    }
    
    // Add new options
    for (const name of Object.keys(members)) {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        memberSelect.appendChild(option);
    }
    
    // Restore previous selection if possible
    if (memberValue && Object.keys(members).includes(memberValue)) {
        memberSelect.value = memberValue;
    }
}

// Calculate distance between two nodes
function calculateLength(node1, node2) {
    const dx = node2.x - node1.x;
    const dy = node2.y - node1.y;
    return Math.sqrt(dx * dx + dy * dy);
}

// Show alert message
function showAlert(type, message) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Find a good place to show the alert
    const container = document.querySelector('main .container') || document.querySelector('main');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 150);
    }, 5000);
}

// Display mock results for demonstration
function displayMockResults() {
    const resultsContainer = document.getElementById('resultsContainer');
    
    let resultsHTML = `
        <h5>Analysis Results</h5>
        <p>The analysis was completed successfully. Here are some sample results:</p>
        
        <div class="mb-3">
            <h6>Degrees of Freedom</h6>
            <ul class="list-group">
    `;
    
    // Create some mock DOFs based on the nodes
    for (const [name, node] of Object.entries(nodes)) {
        if (!node.isSupport) {
            resultsHTML += `
                <li class="list-group-item">Node ${name} - Rotation: ${(Math.random() * 0.01).toFixed(5)} rad</li>
            `;
        }
    }
    
    resultsHTML += `
            </ul>
        </div>
        
        <div class="mb-3">
            <h6>Member End Forces</h6>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Member</th>
                        <th>Start Moment (kNm)</th>
                        <th>End Moment (kNm)</th>
                        <th>Start Shear (kN)</th>
                        <th>End Shear (kN)</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Create mock member results
    for (const [name, member] of Object.entries(members)) {
        resultsHTML += `
            <tr>
                <td>${name}</td>
                <td>${(Math.random() * 20 - 10).toFixed(2)}</td>
                <td>${(Math.random() * 20 - 10).toFixed(2)}</td>
                <td>${(Math.random() * 30 - 15).toFixed(2)}</td>
                <td>${(Math.random() * 30 - 15).toFixed(2)}</td>
            </tr>
        `;
    }
    
    resultsHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    resultsContainer.innerHTML = resultsHTML;
}

// Load continuous beam example
function loadContinuousBeamExample() {
    // Add nodes
    nodes = {
        'A': { x: 0, y: 0, isSupport: true, supportType: 'fixed' },
        'B': { x: 5, y: 0, isSupport: false, supportType: null },
        'C': { x: 10, y: 0, isSupport: false, supportType: null },
        'D': { x: 15, y: 0, isSupport: true, supportType: 'fixed' }
    };
    
    // Add members
    members = {
        'AB': { 
            startNode: 'A', 
            endNode: 'B', 
            EI: 1000, 
            length: calculateLength(nodes['A'], nodes['B'])
        },
        'BC': { 
            startNode: 'B', 
            endNode: 'C', 
            EI: 2000, 
            length: calculateLength(nodes['B'], nodes['C'])
        },
        'CD': { 
            startNode: 'C', 
            endNode: 'D', 
            EI: 1000, 
            length: calculateLength(nodes['C'], nodes['D'])
        }
    };
    
    // Add loads
    loads = {
        'load1': { memberName: 'AB', loadType: 'uniform', magnitude: 20, location: null },
        'load2': { memberName: 'BC', loadType: 'point', magnitude: 80, location: 2 },
        'load3': { memberName: 'CD', loadType: 'uniform', magnitude: 15, location: null }
    };
    
    // Update UI
    updateNodesList();
    updateMembersList();
    updateLoadsList();
    updateNodeSelections();
    updateMemberSelections();
    
    showAlert('success', 'Continuous beam example loaded.');
}

// Load portal frame example
function loadPortalFrameExample() {
    // Add nodes
    nodes = {
        'A': { x: 0, y: 0, isSupport: true, supportType: 'fixed' },
        'B': { x: 0, y: 5, isSupport: false, supportType: null },
        'C': { x: 4, y: 5, isSupport: false, supportType: null },
        'D': { x: 4, y: 0, isSupport: true, supportType: 'fixed' }
    };
    
    // Add members
    members = {
        'AB': { 
            startNode: 'A', 
            endNode: 'B', 
            EI: 1000, 
            length: calculateLength(nodes['A'], nodes['B'])
        },
        'BC': { 
            startNode: 'B', 
            endNode: 'C', 
            EI: 1000, 
            length: calculateLength(nodes['B'], nodes['C'])
        },
        'CD': { 
            startNode: 'C', 
            endNode: 'D', 
            EI: 1000, 
            length: calculateLength(nodes['C'], nodes['D'])
        }
    };
    
    // Add loads
    loads = {
        'load1': { memberName: 'BC', loadType: 'point', magnitude: -50, location: 2 }
    };
    
    // Update UI
    updateNodesList();
    updateMembersList();
    updateLoadsList();
    updateNodeSelections();
    updateMemberSelections();
    
    showAlert('success', 'Portal frame example loaded.');
}