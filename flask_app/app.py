import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from data_utils import split_dataset, get_dataloaders
from models import initialize_vgg19
from train_utils import train_model, evaluate_model, save_plots, save_metrics
from predict_utils import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    dataset_ready = os.path.exists(DATASET_DIR) and os.path.exists(os.path.join(DATASET_DIR, 'train'))
    model_ready = os.path.exists(MODEL_SAVE_PATH)
    return render_template('index.html', dataset_ready=dataset_ready, model_ready=model_ready)


@app.route('/prepare_dataset', methods=['POST'])
def prepare_dataset():
    try:
        if os.path.exists(DATASET_DIR):
            return jsonify({
                'success': True,
                'message': 'Dataset already prepared'
            })
        
        split_dataset(DATA_DIR, DATASET_DIR, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
        
        return jsonify({
            'success': True,
            'message': 'Dataset prepared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        epochs = int(data.get('epochs', EPOCHS))
        learning_rate = float(data.get('learning_rate', LEARNING_RATE))
        batch_size = int(data.get('batch_size', BATCH_SIZE))
        continue_training = data.get('continue_training', False)
        
        if not os.path.exists(DATASET_DIR):
            return jsonify({
                'success': False,
                'error': 'Dataset not prepared. Please prepare dataset first.'
            }), 400
        
        dataloaders, dataset_sizes, class_names = get_dataloaders(DATASET_DIR, batch_size=batch_size)
        
        checkpoint_path = None
        if continue_training and os.path.exists(MODEL_SAVE_PATH):
            checkpoint_path = MODEL_SAVE_PATH
            print(f"Continuing training from checkpoint: {MODEL_SAVE_PATH}")
        else:
            print("Starting fresh training from ImageNet pretrained weights")
        
        model, params_to_update = initialize_vgg19(num_classes=NUM_CLASSES, feature_extract=True, checkpoint_path=checkpoint_path)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=MOMENTUM)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        model, train_loss, train_acc, val_loss, val_acc, train_time = train_model(
            model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=epochs
        )
        
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        metrics = evaluate_model(model, dataloaders['test'], class_names)
        
        save_plots(train_loss, val_loss, train_acc, val_acc, 
                  metrics['confusion_matrix'], class_names, RESULTS_DIR)
        
        save_metrics(metrics, train_time, RESULTS_DIR)
        
        training_mode = "continued from checkpoint" if checkpoint_path else "fresh from ImageNet"
        
        return jsonify({
            'success': True,
            'message': 'Training completed successfully',
            'training_mode': training_mode,
            'train_time': f'{train_time:.2f}',
            'test_time': f"{metrics['test_time']:.2f}",
            'accuracy': f"{metrics['accuracy']:.4f}",
            'precision': f"{metrics['precision']:.4f}",
            'recall': f"{metrics['recall']:.4f}",
            'f1_score': f"{metrics['f1_score']:.4f}",
            'roc_auc': f"{metrics['roc_auc']:.4f}"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image uploaded'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: png, jpg, jpeg'
            }), 400
        
        if not os.path.exists(MODEL_SAVE_PATH):
            return jsonify({
                'success': False,
                'error': 'Model not trained yet. Please train the model first.'
            }), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath, MODEL_SAVE_PATH)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'image_url': f'/uploads/{filename}'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route('/get_metrics')
def get_metrics():
    try:
        metrics_path = os.path.join(RESULTS_DIR, 'metrics.txt')
        if not os.path.exists(metrics_path):
            return jsonify({
                'success': False,
                'error': 'No metrics available. Train the model first.'
            }), 404
        
        with open(metrics_path, 'r') as f:
            metrics_text = f.read()
        
        return jsonify({
            'success': True,
            'metrics': metrics_text
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

