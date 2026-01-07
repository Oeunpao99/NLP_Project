document.addEventListener("DOMContentLoaded", function() {
    // DOM Elements
    const historyList = document.getElementById('historyList');
    const anonHistoryList = document.getElementById('anonHistoryList');
    const emptyHistory = document.getElementById('emptyHistory');
    const userHistory = document.getElementById('userHistory');
    const anonymousHistory = document.getElementById('anonymousHistory');
    const totalPredictions = document.getElementById('totalPredictions');
    const searchInput = document.getElementById('searchInput');
    const dateFilter = document.getElementById('dateFilter');
    const prevPage = document.getElementById('prevPage');
    const nextPage = document.getElementById('nextPage');
    const pageInfo = document.getElementById('pageInfo');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const detailModal = document.getElementById('detailModal');
    const closeModal = document.querySelector('.close-modal');
    const reuseTextBtn = document.getElementById('reuseTextBtn');
    const exportJsonBtn = document.getElementById('exportJsonBtn');
    
    // State
    let currentPage = 1;
    let totalPages = 1;
    let currentPredictions = [];
    let currentDetail = null;
    
    // Initialize
    loadHistory();
    
    // Load history based on authentication
    async function loadHistory(page = 1) {
        const token = localStorage.getItem('access_token');
        currentPage = page;
        
        if (token) {
            // Load from API (logged in user)
            await loadUserHistory(page);
            userHistory.style.display = 'block';
            anonymousHistory.style.display = 'none';
        } else {
            // Load from localStorage (anonymous user)
            loadAnonymousHistory();
            userHistory.style.display = 'none';
            anonymousHistory.style.display = 'block';
        }
        
        updateEmptyState();
    }
    
    // Load user history from API
    async function loadUserHistory(page = 1) {
        try {
            showLoadingSkeleton();
            
            const token = localStorage.getItem('access_token');
            const response = await fetch(`/api/v1/predictions?page=${page}&page_size=10`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                currentPredictions = data.predictions || [];
                
                renderHistoryList(currentPredictions);
                updatePagination(page, data);
                totalPredictions.textContent = `${data.total} predictions`;
                
            } else if (response.status === 401) {
                // Token expired, clear and reload as anonymous
                localStorage.removeItem('access_token');
                loadHistory();
            } else {
                console.error('Failed to load history:', await response.text());
                historyList.innerHTML = '<div class="alert alert-error">Failed to load history</div>';
            }
        } catch (error) {
            console.error('Error loading history:', error);
            historyList.innerHTML = '<div class="alert alert-error">Connection error</div>';
        }
    }
    
    // Load anonymous history from localStorage
    function loadAnonymousHistory() {
        try {
            const history = JSON.parse(localStorage.getItem('anon_history') || '[]');
            currentPredictions = history;
            
            renderAnonymousHistoryList(history);
            totalPredictions.textContent = `${history.length} predictions`;
            
            // For anonymous users, show simple pagination
            updateSimplePagination(history.length);
            
        } catch (error) {
            console.error('Error loading anonymous history:', error);
            anonHistoryList.innerHTML = '<div class="alert alert-error">Error loading history</div>';
        }
    }
    
    // Render user history list
    function renderHistoryList(predictions) {
        historyList.innerHTML = '';
        
        if (predictions.length === 0) {
            historyList.innerHTML = '<div class="empty-list">No predictions found</div>';
            return;
        }
        
        const template = document.getElementById('historyItemTemplate');
        
        predictions.forEach(prediction => {
            const clone = template.content.cloneNode(true);
            
            // Format date
            const date = new Date(prediction.created_at);
            clone.querySelector('.date').textContent = formatDate(date);
            clone.querySelector('.time').textContent = formatTime(date);
            
            // Count entities
            const entityCount = prediction.entities ? 
                Object.values(prediction.entities).reduce((sum, arr) => sum + (arr?.length || 0), 0) : 0;
            clone.querySelector('.entity-count').textContent = entityCount;
            clone.querySelector('.inference-time').textContent = 
                `${prediction.inference_time_ms?.toFixed(1) || 0}ms`;
            
            // Truncate text
            const text = prediction.text || '';
            const truncatedText = text.length > 150 ? text.substring(0, 150) + '...' : text;
            const textElement = clone.querySelector('.history-item-text');
            textElement.textContent = truncatedText;
            
            // Add expand button for long text
            if (text.length > 150) {
                const expandBtn = document.createElement('button');
                expandBtn.className = 'expand-btn';
                expandBtn.textContent = 'Show more';
                expandBtn.addEventListener('click', () => {
                    if (textElement.textContent.length > 150) {
                        textElement.textContent = text;
                        expandBtn.textContent = 'Show less';
                    } else {
                        textElement.textContent = truncatedText;
                        expandBtn.textContent = 'Show more';
                    }
                });
                textElement.parentNode.appendChild(expandBtn);
            }
            
            // Add click handlers
            const viewBtn = clone.querySelector('.btn-view');
            const deleteBtn = clone.querySelector('.btn-delete');
            
            viewBtn.addEventListener('click', () => showDetailModal(prediction));
            deleteBtn.addEventListener('click', () => deletePrediction(prediction.id));
            
            historyList.appendChild(clone);
        });
    }
    
    // Render anonymous history list
    function renderAnonymousHistoryList(predictions) {
        anonHistoryList.innerHTML = '';
        
        if (predictions.length === 0) {
            anonHistoryList.innerHTML = '<div class="empty-list">No predictions found</div>';
            return;
        }
        
        const template = document.getElementById('historyItemTemplate');
        
        // Show only first 50 items for anonymous users
        const displayItems = predictions.slice(0, 50);
        
        displayItems.forEach((prediction, index) => {
            const clone = template.content.cloneNode(true);
            
            const date = new Date(prediction.created_at);
            clone.querySelector('.date').textContent = formatDate(date);
            clone.querySelector('.time').textContent = formatTime(date);
            
            const entityCount = prediction.entities ? 
                Object.values(prediction.entities).reduce((sum, arr) => sum + (arr?.length || 0), 0) : 0;
            clone.querySelector('.entity-count').textContent = entityCount;
            clone.querySelector('.inference-time').textContent = 
                `${prediction.inference_time_ms?.toFixed(1) || 0}ms`;
            
            const text = prediction.text || '';
            const truncatedText = text.length > 150 ? text.substring(0, 150) + '...' : text;
            clone.querySelector('.history-item-text').textContent = truncatedText;
            
            const viewBtn = clone.querySelector('.btn-view');
            const deleteBtn = clone.querySelector('.btn-delete');
            
            viewBtn.addEventListener('click', () => showDetailModal(prediction));
            deleteBtn.addEventListener('click', () => deleteAnonymousPrediction(index));
            
            anonHistoryList.appendChild(clone);
        });
    }
    
    // Show detail modal
    function showDetailModal(prediction) {
        currentDetail = prediction;
        
        const date = new Date(prediction.created_at);
        document.getElementById('detailDateTime').textContent = 
            `${formatDate(date)} ${formatTime(date)}`;
        
        document.getElementById('detailText').textContent = prediction.text || '';
        document.getElementById('detailTextLength').textContent = prediction.text?.length || 0;
        document.getElementById('detailInferenceTime').textContent = 
            `${prediction.inference_time_ms?.toFixed(1) || 0}ms`;
        document.getElementById('detailModelVersion').textContent = 
            prediction.model_version || 'v1.0';
        
        // Display entities
        const entitiesContainer = document.getElementById('detailEntities');
        entitiesContainer.innerHTML = '';
        
        if (prediction.entities && typeof prediction.entities === 'object') {
            Object.entries(prediction.entities).forEach(([type, entities]) => {
                if (entities && entities.length > 0) {
                    const typeDiv = document.createElement('div');
                    typeDiv.className = 'entity-type-group';
                    typeDiv.innerHTML = `
                        <h5>${type} (${entities.length})</h5>
                        <div class="entity-list">
                            ${entities.map(entity => 
                                `<span class="entity-badge">${entity}</span>`
                            ).join('')}
                        </div>
                    `;
                    entitiesContainer.appendChild(typeDiv);
                }
            });
        }
        
        detailModal.style.display = 'block';
    }
    
    // Delete prediction (authenticated users)
    async function deletePrediction(predictionId) {
        if (!confirm('Are you sure you want to delete this prediction?')) return;
        
        try {
            const token = localStorage.getItem('access_token');
            const response = await fetch(`/api/v1/predictions/${predictionId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            
            if (response.ok) {
                // Reload current page
                loadHistory(currentPage);
            } else {
                const error = await response.text();
                alert(`Failed to delete: ${error}`);
            }
        } catch (error) {
            console.error('Error deleting prediction:', error);
            alert('Failed to delete prediction');
        }
    }
    
    // Delete anonymous prediction
    function deleteAnonymousPrediction(index) {
        if (!confirm('Are you sure you want to delete this prediction?')) return;
        
        try {
            const history = JSON.parse(localStorage.getItem('anon_history') || '[]');
            history.splice(index, 1);
            localStorage.setItem('anon_history', JSON.stringify(history));
            loadAnonymousHistory();
            updateEmptyState();
        } catch (error) {
            console.error('Error deleting anonymous prediction:', error);
            alert('Failed to delete prediction');
        }
    }
    
    // Clear all history
    async function clearAllHistory() {
        if (!confirm('Are you sure you want to clear ALL history? This cannot be undone.')) return;
        
        const token = localStorage.getItem('access_token');
        
        if (token) {
            try {
                // You need to implement a DELETE /api/v1/predictions/all endpoint
                // Or delete one by one
                alert('Feature coming soon: Bulk delete for authenticated users');
            } catch (error) {
                alert('Failed to clear history');
            }
        } else {
            // Clear localStorage
            localStorage.removeItem('anon_history');
            loadAnonymousHistory();
            updateEmptyState();
        }
    }
    
    // Update pagination controls
    function updatePagination(page, data) {
        currentPage = page;
        totalPages = data.total_pages || 1;
        
        prevPage.disabled = !data.has_prev;
        nextPage.disabled = !data.has_next;
        pageInfo.textContent = `Page ${page} of ${totalPages}`;
    }
    
    // Simple pagination for anonymous users
    function updateSimplePagination(totalItems) {
        const itemsPerPage = 10;
        totalPages = Math.ceil(totalItems / itemsPerPage);
        
        prevPage.disabled = currentPage <= 1;
        nextPage.disabled = currentPage >= totalPages;
        pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
    }
    
    // Update empty state
    function updateEmptyState() {
        const token = localStorage.getItem('access_token');
        let hasHistory = false;
        
        if (token) {
            hasHistory = currentPredictions.length > 0;
        } else {
            const history = JSON.parse(localStorage.getItem('anon_history') || '[]');
            hasHistory = history.length > 0;
        }
        
        if (hasHistory) {
            emptyHistory.style.display = 'none';
        } else {
            emptyHistory.style.display = 'block';
        }
    }
    
    // Show loading skeleton
    function showLoadingSkeleton(count = 3) {
        historyList.innerHTML = '';
        
        for (let i = 0; i < count; i++) {
            const skeleton = document.createElement('div');
            skeleton.className = 'history-item loading-item';
            skeleton.innerHTML = `
                <div style="height: 20px; width: 60%; background: #e2e8f0; margin-bottom: 1rem; border-radius: 4px;"></div>
                <div style="height: 15px; width: 40%; background: #e2e8f0; margin-bottom: 1rem; border-radius: 4px;"></div>
                <div style="height: 60px; background: #e2e8f0; margin-bottom: 1rem; border-radius: 4px;"></div>
                <div style="display: flex; gap: 0.5rem;">
                    <div style="height: 30px; width: 80px; background: #e2e8f0; border-radius: 4px;"></div>
                    <div style="height: 30px; width: 80px; background: #e2e8f0; border-radius: 4px;"></div>
                </div>
            `;
            historyList.appendChild(skeleton);
        }
    }
    
    // Format date
    function formatDate(date) {
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }
    
    // Format time
    function formatTime(date) {
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    // Event Listeners
    prevPage.addEventListener('click', () => {
        if (currentPage > 1) {
            loadHistory(currentPage - 1);
        }
    });
    
    nextPage.addEventListener('click', () => {
        if (currentPage < totalPages) {
            loadHistory(currentPage + 1);
        }
    });
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        // Implement search filtering
        filterHistory(searchTerm);
    });
    
    dateFilter.addEventListener('change', function() {
        filterByDate(this.value);
    });
    
    clearHistoryBtn.addEventListener('click', clearAllHistory);
    
    logoutBtn.addEventListener('click', function() {
        localStorage.removeItem('access_token');
        window.location.href = '/';
    });
    
    closeModal.addEventListener('click', function() {
        detailModal.style.display = 'none';
    });
    
    reuseTextBtn.addEventListener('click', function() {
        if (currentDetail && currentDetail.text) {
            localStorage.setItem('reuse_text', currentDetail.text);
            window.location.href = '/';
        }
    });
    
    exportJsonBtn.addEventListener('click', function() {
        if (currentDetail) {
            const dataStr = JSON.stringify(currentDetail, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
            const exportFileDefaultName = `khmer-ner-prediction-${Date.now()}.json`;
            
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', exportFileDefaultName);
            linkElement.click();
        }
    });
    
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === detailModal) {
            detailModal.style.display = 'none';
        }
    });
    
    // Filter functions
    function filterHistory(searchTerm) {
        // Implement search filtering
    }
    
    function filterByDate(filterValue) {
        // Implement date filtering
    }
});
async function loadUserHistory(page = 1) {
    try {
        showLoadingSkeleton();
        
        const token = localStorage.getItem('access_token');
        const headers = {};
        
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        const response = await fetch(`/api/v1/predictions?page=${page}&page_size=10`, {
            headers: headers
        });
        
        if (response.ok) {
            const data = await response.json();
            currentPredictions = data.predictions || [];
            
            renderHistoryList(currentPredictions);
            updatePagination(data);
            totalPredictions.textContent = `${data.total} predictions`;
            
        } else if (response.status === 401) {
            // Token expired, show anonymous history
            console.log("Not authenticated, showing anonymous history");
            loadAnonymousHistory();
        } else {
            console.error('Failed to load history:', await response.text());
            historyList.innerHTML = '<div class="alert alert-error">Failed to load history</div>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        historyList.innerHTML = '<div class="alert alert-error">Connection error</div>';
    }
}