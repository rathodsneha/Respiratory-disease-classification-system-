from flask import render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
import os
import json
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from app.dashboard import bp
from app.auth.routes import permission_required
from app.patient.models import Patient, AudioFile, ModelPrediction, RecordingSession
from app import db

@bp.route('/')
@login_required
def index():
    """Main dashboard"""
    
    # Get dashboard statistics
    stats = get_dashboard_stats()
    
    # Get recent activity
    recent_activity = get_recent_activity()
    
    # Get model performance summary
    model_performance = get_model_performance_summary()
    
    return render_template('dashboard/index.html', 
                         title='Dashboard',
                         stats=stats,
                         recent_activity=recent_activity,
                         model_performance=model_performance)

@bp.route('/compare-models')
@login_required
@permission_required('can_run_models')
def compare_models():
    """Model comparison dashboard"""
    
    # Get all audio files with predictions
    audio_files = AudioFile.query.filter_by(is_processed=True).all()
    
    # Group predictions by model
    model_comparisons = {}
    for audio_file in audio_files:
        predictions = audio_file.predictions.all()
        for pred in predictions:
            if pred.model_name not in model_comparisons:
                model_comparisons[pred.model_name] = []
            model_comparisons[pred.model_name].append(pred)
    
    # Generate comparison charts
    comparison_charts = generate_model_comparison_charts(model_comparisons)
    
    return render_template('dashboard/compare_models.html',
                         title='Model Comparison',
                         model_comparisons=model_comparisons,
                         comparison_charts=comparison_charts)

@bp.route('/single-audio-analysis/<int:audio_id>')
@login_required
@permission_required('can_run_models')
def single_audio_analysis(audio_id):
    """Single audio file analysis with all three models"""
    
    audio_file = AudioFile.query.get_or_404(audio_id)
    
    # Get predictions from all models
    predictions = audio_file.predictions.all()
    model_predictions = {}
    for pred in predictions:
        model_predictions[pred.model_name] = pred
    
    # Generate visualizations
    analysis_charts = generate_single_audio_charts(audio_file, model_predictions)
    
    return render_template('dashboard/single_audio_analysis.html',
                         title=f'Audio Analysis - {audio_file.filename}',
                         audio_file=audio_file,
                         model_predictions=model_predictions,
                         analysis_charts=analysis_charts)

@bp.route('/batch-analysis')
@login_required
@permission_required('can_run_models')
def batch_analysis():
    """Batch audio analysis dashboard"""
    
    # Get recent batch analyses
    recent_batches = get_recent_batch_analyses()
    
    # Get batch performance statistics
    batch_stats = get_batch_performance_stats()
    
    return render_template('dashboard/batch_analysis.html',
                         title='Batch Analysis',
                         recent_batches=recent_batches,
                         batch_stats=batch_stats)

@bp.route('/performance-metrics')
@login_required
@permission_required('can_view_reports')
def performance_metrics():
    """Detailed performance metrics dashboard"""
    
    # Get performance metrics for all models
    metrics = get_detailed_performance_metrics()
    
    # Generate performance charts
    performance_charts = generate_performance_charts(metrics)
    
    return render_template('dashboard/performance_metrics.html',
                         title='Performance Metrics',
                         metrics=metrics,
                         performance_charts=performance_charts)

@bp.route('/api/model-comparison-data')
@login_required
@permission_required('can_run_models')
def api_model_comparison_data():
    """API endpoint for model comparison data"""
    
    # Get comparison data
    data = get_model_comparison_data()
    
    return jsonify(data)

@bp.route('/api/audio-analysis/<int:audio_id>')
@login_required
@permission_required('can_run_models')
def api_audio_analysis(audio_id):
    """API endpoint for single audio analysis data"""
    
    audio_file = AudioFile.query.get_or_404(audio_id)
    predictions = audio_file.predictions.all()
    
    analysis_data = {
        'audio_file': {
            'id': audio_file.id,
            'filename': audio_file.filename,
            'duration': audio_file.duration,
            'format': audio_file.format
        },
        'predictions': []
    }
    
    for pred in predictions:
        pred_data = {
            'model_name': pred.model_name,
            'predicted_class': pred.predicted_class,
            'confidence_score': pred.confidence_score,
            'class_probabilities': json.loads(pred.class_probabilities) if pred.class_probabilities else {},
            'risk_level': pred.risk_level,
            'processing_time': pred.processing_time
        }
        analysis_data['predictions'].append(pred_data)
    
    return jsonify(analysis_data)

# Helper functions

def get_dashboard_stats():
    """Get dashboard statistics"""
    stats = {
        'total_patients': Patient.query.count(),
        'total_recordings': AudioFile.query.count(),
        'processed_recordings': AudioFile.query.filter_by(is_processed=True).count(),
        'total_predictions': ModelPrediction.query.count(),
        'active_sessions': RecordingSession.query.filter(
            RecordingSession.session_date >= datetime.utcnow() - timedelta(days=30)
        ).count()
    }
    
    # Calculate processing rate
    if stats['total_recordings'] > 0:
        stats['processing_rate'] = (stats['processed_recordings'] / stats['total_recordings']) * 100
    else:
        stats['processing_rate'] = 0
    
    return stats

def get_recent_activity():
    """Get recent system activity"""
    recent_predictions = ModelPrediction.query.order_by(
        ModelPrediction.predicted_at.desc()
    ).limit(10).all()
    
    activity = []
    for pred in recent_predictions:
        activity.append({
            'timestamp': pred.predicted_at,
            'type': 'prediction',
            'description': f'{pred.model_name} predicted {pred.predicted_class} with {pred.confidence_score:.1%} confidence',
            'audio_file': pred.audio_file.filename if pred.audio_file else 'Unknown'
        })
    
    return activity

def get_model_performance_summary():
    """Get model performance summary"""
    models = ['CNN', 'LSTM', 'Hybrid']
    performance = {}
    
    for model in models:
        predictions = ModelPrediction.query.filter_by(model_name=model).all()
        if predictions:
            confidence_scores = [p.confidence_score for p in predictions]
            avg_confidence = np.mean(confidence_scores)
            avg_processing_time = np.mean([p.processing_time for p in predictions if p.processing_time])
            
            performance[model] = {
                'total_predictions': len(predictions),
                'avg_confidence': avg_confidence,
                'avg_processing_time': avg_processing_time,
                'last_prediction': max(p.predicted_at for p in predictions) if predictions else None
            }
        else:
            performance[model] = {
                'total_predictions': 0,
                'avg_confidence': 0,
                'avg_processing_time': 0,
                'last_prediction': None
            }
    
    return performance

def generate_model_comparison_charts(model_comparisons):
    """Generate model comparison charts"""
    charts = {}
    
    # Confidence score comparison
    confidence_data = []
    for model_name, predictions in model_comparisons.items():
        confidence_scores = [p.confidence_score for p in predictions]
        confidence_data.append(go.Box(
            y=confidence_scores,
            name=model_name,
            boxpoints='outliers'
        ))
    
    confidence_fig = go.Figure(data=confidence_data)
    confidence_fig.update_layout(
        title='Model Confidence Score Comparison',
        yaxis_title='Confidence Score',
        xaxis_title='Model'
    )
    charts['confidence_comparison'] = json.dumps(confidence_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Processing time comparison
    time_data = []
    for model_name, predictions in model_comparisons.items():
        processing_times = [p.processing_time for p in predictions if p.processing_time]
        if processing_times:
            time_data.append(go.Box(
                y=processing_times,
                name=model_name,
                boxpoints='outliers'
            ))
    
    if time_data:
        time_fig = go.Figure(data=time_data)
        time_fig.update_layout(
            title='Model Processing Time Comparison',
            yaxis_title='Processing Time (seconds)',
            xaxis_title='Model'
        )
        charts['processing_time_comparison'] = json.dumps(time_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Disease distribution by model
    disease_data = {}
    for model_name, predictions in model_comparisons.items():
        disease_counts = {}
        for pred in predictions:
            disease = pred.predicted_class
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        disease_data[model_name] = disease_counts
    
    # Create disease distribution chart
    diseases = set()
    for model_data in disease_data.values():
        diseases.update(model_data.keys())
    diseases = list(diseases)
    
    disease_fig_data = []
    for model_name, disease_counts in disease_data.items():
        counts = [disease_counts.get(disease, 0) for disease in diseases]
        disease_fig_data.append(go.Bar(
            x=diseases,
            y=counts,
            name=model_name
        ))
    
    disease_fig = go.Figure(data=disease_fig_data)
    disease_fig.update_layout(
        title='Disease Classification Distribution by Model',
        xaxis_title='Disease Class',
        yaxis_title='Number of Predictions',
        barmode='group'
    )
    charts['disease_distribution'] = json.dumps(disease_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return charts

def generate_single_audio_charts(audio_file, model_predictions):
    """Generate charts for single audio analysis"""
    charts = {}
    
    if not model_predictions:
        return charts
    
    # Model confidence comparison for this audio
    models = list(model_predictions.keys())
    confidences = [model_predictions[model].confidence_score for model in models]
    
    confidence_fig = go.Figure(data=[
        go.Bar(x=models, y=confidences, 
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
    ])
    confidence_fig.update_layout(
        title=f'Model Confidence Comparison - {audio_file.filename}',
        xaxis_title='Model',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 1])
    )
    charts['confidence_comparison'] = json.dumps(confidence_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Class probabilities heatmap
    if any(pred.class_probabilities for pred in model_predictions.values()):
        prob_data = []
        class_names = []
        
        for model, pred in model_predictions.items():
            if pred.class_probabilities:
                probs = json.loads(pred.class_probabilities)
                if not class_names:
                    class_names = list(probs.keys())
                prob_data.append([probs.get(cls, 0) for cls in class_names])
        
        if prob_data:
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=prob_data,
                x=class_names,
                y=models,
                colorscale='Viridis',
                showscale=True
            ))
            heatmap_fig.update_layout(
                title='Class Probability Heatmap',
                xaxis_title='Disease Class',
                yaxis_title='Model'
            )
            charts['probability_heatmap'] = json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return charts

def get_model_comparison_data():
    """Get model comparison data for API"""
    models = ['CNN', 'LSTM', 'Hybrid']
    comparison_data = {}
    
    for model in models:
        predictions = ModelPrediction.query.filter_by(model_name=model).all()
        
        if predictions:
            # Calculate metrics
            confidence_scores = [p.confidence_score for p in predictions]
            processing_times = [p.processing_time for p in predictions if p.processing_time]
            
            # Disease distribution
            disease_counts = {}
            for pred in predictions:
                disease = pred.predicted_class
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            comparison_data[model] = {
                'total_predictions': len(predictions),
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'std_confidence': np.std(confidence_scores) if confidence_scores else 0,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'disease_distribution': disease_counts,
                'confidence_distribution': {
                    'high': len([c for c in confidence_scores if c >= 0.8]),
                    'medium': len([c for c in confidence_scores if 0.5 <= c < 0.8]),
                    'low': len([c for c in confidence_scores if c < 0.5])
                }
            }
        else:
            comparison_data[model] = {
                'total_predictions': 0,
                'avg_confidence': 0,
                'std_confidence': 0,
                'avg_processing_time': 0,
                'disease_distribution': {},
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
    
    return comparison_data

def get_recent_batch_analyses():
    """Get recent batch analyses"""
    # This would be implemented based on batch processing logs
    # For now, return mock data
    return []

def get_batch_performance_stats():
    """Get batch performance statistics"""
    # This would be implemented based on batch processing results
    # For now, return mock data
    return {}

def get_detailed_performance_metrics():
    """Get detailed performance metrics"""
    models = ['CNN', 'LSTM', 'Hybrid']
    metrics = {}
    
    for model in models:
        predictions = ModelPrediction.query.filter_by(model_name=model).all()
        
        if predictions:
            # Calculate detailed metrics
            confidence_scores = [p.confidence_score for p in predictions]
            processing_times = [p.processing_time for p in predictions if p.processing_time]
            
            metrics[model] = {
                'accuracy_estimate': np.mean([1 if c > 0.7 else 0 for c in confidence_scores]),
                'precision_estimate': np.mean([c for c in confidence_scores if c > 0.5]),
                'recall_estimate': len([c for c in confidence_scores if c > 0.5]) / len(confidence_scores),
                'f1_score_estimate': 2 * (np.mean([c for c in confidence_scores if c > 0.5]) * 
                                        (len([c for c in confidence_scores if c > 0.5]) / len(confidence_scores))) / 
                                   (np.mean([c for c in confidence_scores if c > 0.5]) + 
                                    (len([c for c in confidence_scores if c > 0.5]) / len(confidence_scores))),
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'total_predictions': len(predictions)
            }
        else:
            metrics[model] = {
                'accuracy_estimate': 0,
                'precision_estimate': 0,
                'recall_estimate': 0,
                'f1_score_estimate': 0,
                'avg_processing_time': 0,
                'total_predictions': 0
            }
    
    return metrics

def generate_performance_charts(metrics):
    """Generate performance charts"""
    charts = {}
    
    models = list(metrics.keys())
    
    # Performance metrics comparison
    metrics_names = ['accuracy_estimate', 'precision_estimate', 'recall_estimate', 'f1_score_estimate']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    performance_fig_data = []
    for i, metric in enumerate(metrics_names):
        values = [metrics[model][metric] for model in models]
        performance_fig_data.append(go.Bar(
            x=models,
            y=values,
            name=metric_labels[i]
        ))
    
    performance_fig = go.Figure(data=performance_fig_data)
    performance_fig.update_layout(
        title='Model Performance Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1])
    )
    charts['performance_metrics'] = json.dumps(performance_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return charts