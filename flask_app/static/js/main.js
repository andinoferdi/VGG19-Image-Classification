document.addEventListener('DOMContentLoaded', function() {
    const prepareDatasetBtn = document.getElementById('prepareDatasetBtn');
    const trainForm = document.getElementById('trainForm');
    const predictForm = document.getElementById('predictForm');
    const imageFile = document.getElementById('imageFile');
    const loadMetricsBtn = document.getElementById('loadMetricsBtn');
    
    if (prepareDatasetBtn) {
        prepareDatasetBtn.addEventListener('click', prepareDataset);
    }
    
    trainForm.addEventListener('submit', handleTraining);
    predictForm.addEventListener('submit', handlePrediction);
    imageFile.addEventListener('change', previewImage);
    loadMetricsBtn.addEventListener('click', loadMetrics);
});

async function prepareDataset() {
    const btn = document.getElementById('prepareDatasetBtn');
    const statusText = document.getElementById('datasetStatusText');
    
    btn.disabled = true;
    btn.textContent = 'Preparing...';
    
    try {
        const response = await fetch('/prepare_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusText.innerHTML = '<span class="text-green-600">Ready</span>';
            btn.remove();
            alert('Dataset prepared successfully!');
        } else {
            alert('Error: ' + data.error);
            btn.disabled = false;
            btn.textContent = 'Prepare Dataset';
        }
    } catch (error) {
        alert('Error preparing dataset: ' + error.message);
        btn.disabled = false;
        btn.textContent = 'Prepare Dataset';
    }
}

async function handleTraining(e) {
    e.preventDefault();
    
    const epochs = document.getElementById('epochs').value;
    const learning_rate = document.getElementById('learning_rate').value;
    const batch_size = document.getElementById('batch_size').value;
    const continue_training = document.getElementById('continue_training').checked;
    
    const trainBtn = document.getElementById('trainBtn');
    const trainingStatus = document.getElementById('trainingStatus');
    const trainingResults = document.getElementById('trainingResults');
    const errorMsg = document.getElementById('errorMsg');
    
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';
    trainingStatus.classList.remove('hidden');
    trainingResults.classList.add('hidden');
    errorMsg.classList.add('hidden');
    
    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                epochs: parseInt(epochs),
                learning_rate: parseFloat(learning_rate),
                batch_size: parseInt(batch_size),
                continue_training: continue_training
            })
        });
        
        const data = await response.json();
        
        trainingStatus.classList.add('hidden');
        
        if (data.success) {
            document.getElementById('trainTime').textContent = data.train_time;
            document.getElementById('testTime').textContent = data.test_time;
            document.getElementById('accuracy').textContent = data.accuracy;
            document.getElementById('precision').textContent = data.precision;
            document.getElementById('recall').textContent = data.recall;
            document.getElementById('f1Score').textContent = data.f1_score;
            document.getElementById('rocAuc').textContent = data.roc_auc;
            trainingResults.classList.remove('hidden');
        } else {
            document.getElementById('errorText').textContent = data.error;
            errorMsg.classList.remove('hidden');
        }
    } catch (error) {
        trainingStatus.classList.add('hidden');
        document.getElementById('errorText').textContent = 'Error: ' + error.message;
        errorMsg.classList.remove('hidden');
    } finally {
        trainBtn.disabled = false;
        trainBtn.textContent = 'Start Training';
    }
}

function previewImage() {
    const file = document.getElementById('imageFile').files[0];
    const preview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            preview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        preview.classList.add('hidden');
    }
}

async function handlePrediction(e) {
    e.preventDefault();
    
    const imageFile = document.getElementById('imageFile').files[0];
    
    if (!imageFile) {
        alert('Please select an image file');
        return;
    }
    
    const predictBtn = document.getElementById('predictBtn');
    const predictionStatus = document.getElementById('predictionStatus');
    const predictionResults = document.getElementById('predictionResults');
    const predictError = document.getElementById('predictError');
    
    predictBtn.disabled = true;
    predictBtn.textContent = 'Processing...';
    predictionStatus.classList.remove('hidden');
    predictionResults.classList.add('hidden');
    predictError.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('image', imageFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        predictionStatus.classList.add('hidden');
        
        if (data.success) {
            const prediction = data.prediction;
            
            document.getElementById('predictedClass').textContent = prediction.predicted_class;
            document.getElementById('confidence').textContent = (prediction.confidence * 100).toFixed(2) + '%';
            
            const top3Container = document.getElementById('top3Predictions');
            top3Container.innerHTML = '';
            
            prediction.top_3.forEach((item, index) => {
                const div = document.createElement('div');
                div.className = 'flex justify-between text-sm';
                div.innerHTML = `
                    <span class="text-gray-700">${index + 1}. ${item.class}</span>
                    <span class="text-gray-600 font-medium">${(item.confidence * 100).toFixed(2)}%</span>
                `;
                top3Container.appendChild(div);
            });
            
            predictionResults.classList.remove('hidden');
        } else {
            document.getElementById('predictErrorText').textContent = data.error;
            predictError.classList.remove('hidden');
        }
    } catch (error) {
        predictionStatus.classList.add('hidden');
        document.getElementById('predictErrorText').textContent = 'Error: ' + error.message;
        predictError.classList.remove('hidden');
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict';
    }
}

async function loadMetrics() {
    const visualizations = document.getElementById('visualizations');
    const metricsText = document.getElementById('metricsText');
    const historyPlot = document.getElementById('historyPlot');
    const confusionMatrix = document.getElementById('confusionMatrix');
    
    try {
        const response = await fetch('/get_metrics');
        const data = await response.json();
        
        if (data.success) {
            const timestamp = new Date().getTime();
            historyPlot.src = `/results/training_history.png?t=${timestamp}`;
            confusionMatrix.src = `/results/confusion_matrix.png?t=${timestamp}`;
            
            document.getElementById('metricsContent').textContent = data.metrics;
            
            visualizations.classList.remove('hidden');
            metricsText.classList.remove('hidden');
        } else {
            alert('Error loading metrics: ' + data.error);
        }
    } catch (error) {
        alert('Error loading metrics: ' + error.message);
    }
}

