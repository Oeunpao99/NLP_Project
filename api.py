# # app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# import numpy as np
# import re
# from model import BiLSTM_CRF  # Your model class
# import pickle

# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend

# # Load model and preprocessing
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load vocabulary and tag mapping
# with open('vocab.pkl', 'rb') as f:
#     word2idx = pickle.load(f)
# with open('tag2idx.pkl', 'rb') as f:
#     tag2idx = pickle.load(f)
# with open('idx2tag.pkl', 'rb') as f:
#     idx2tag = pickle.load(f)

# # Initialize model
# vocab_size = len(word2idx)
# tagset_size = len(tag2idx)
# embedding_dim = 100
# hidden_dim = 256

# model = BiLSTM_CRF(vocab_size, tagset_size, embedding_dim, hidden_dim)
# model.load_state_dict(torch.load('bilstm_crf_best.pt', map_location=device))
# model.to(device)
# model.eval()

# def preprocess_khmer_text(text):
#     """Preprocess Khmer text for model input"""
#     # Tokenize Khmer text (simple space-based tokenization)
#     tokens = text.split()
    
#     # Convert tokens to indices
#     token_indices = []
#     for token in tokens:
#         # Handle unknown words
#         idx = word2idx.get(token, word2idx.get('<UNK>', 0))
#         token_indices.append(idx)
    
#     return tokens, torch.tensor([token_indices], dtype=torch.long)

# def decode_predictions(tokens, predictions):
#     """Convert model predictions to entities"""
#     entities = []
#     current_entity = None
#     entity_start = 0
    
#     for i, (token, tag_idx) in enumerate(zip(tokens, predictions)):
#         tag = idx2tag[tag_idx]
        
#         # Check for B- tag (beginning of entity)
#         if tag.startswith('B-'):
#             if current_entity:
#                 # Save previous entity
#                 entity_text = ' '.join(tokens[entity_start:i])
#                 entities.append({
#                     'text': entity_text,
#                     'type': current_entity,
#                     'start': entity_start,
#                     'end': i
#                 })
            
#             # Start new entity
#             current_entity = tag[2:]  # Remove B- prefix
#             entity_start = i
        
#         # Check for I- tag (inside entity)
#         elif tag.startswith('I-'):
#             # Continue current entity
#             if current_entity and current_entity == tag[2:]:
#                 continue
#             else:
#                 # Mismatch, end current entity
#                 if current_entity:
#                     entity_text = ' '.join(tokens[entity_start:i])
#                     entities.append({
#                         'text': entity_text,
#                         'type': current_entity,
#                         'start': entity_start,
#                         'end': i
#                     })
#                     current_entity = None
        
#         # O tag (outside entity)
#         else:
#             if current_entity:
#                 # End current entity
#                 entity_text = ' '.join(tokens[entity_start:i])
#                 entities.append({
#                     'text': entity_text,
#                     'type': current_entity,
#                     'start': entity_start,
#                     'end': i
#                 })
#                 current_entity = None
    
#     # Handle last entity if exists
#     if current_entity:
#         entity_text = ' '.join(tokens[entity_start:len(tokens)])
#         entities.append({
#             'text': entity_text,
#             'type': current_entity,
#             'start': entity_start,
#             'end': len(tokens)
#         })
    
#     return entities

# @app.route('/api/analyze', methods=['POST'])
# def analyze_text():
#     try:
#         data = request.get_json()
#         text = data.get('text', '')
        
#         if not text:
#             return jsonify({'error': 'No text provided'}), 400
        
#         # Preprocess text
#         tokens, input_tensor = preprocess_khmer_text(text)
        
#         # Make prediction
#         with torch.no_grad():
#             input_tensor = input_tensor.to(device)
#             mask = torch.ones_like(input_tensor, dtype=torch.uint8).to(device)
#             predictions = model(input_tensor, mask)
        
#         # Decode predictions
#         entities = decode_predictions(tokens, predictions[0])
        
#         # Convert entity types to match frontend
#         entity_type_mapping = {
#             'PER': 'person',
#             'LOC': 'location',
#             'ORG': 'organization',
#             'DATE': 'date',
#             'MONEY': 'money'
#         }
        
#         for entity in entities:
#             if entity['type'] in entity_type_mapping:
#                 entity['type'] = entity_type_mapping[entity['type']]
        
#         # Calculate statistics
#         stats = {
#             'person': sum(1 for e in entities if e['type'] == 'person'),
#             'location': sum(1 for e in entities if e['type'] == 'location'),
#             'organization': sum(1 for e in entities if e['type'] == 'organization'),
#             'date': sum(1 for e in entities if e['type'] == 'date'),
#             'money': sum(1 for e in entities if e['type'] == 'money'),
#             'total': len(entities)
#         }
        
#         return jsonify({
#             'success': True,
#             'text': text,
#             'entities': entities,
#             'stats': stats,
#             'tokens': tokens
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy', 'model': 'loaded'})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)