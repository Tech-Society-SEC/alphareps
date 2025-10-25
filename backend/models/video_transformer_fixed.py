#!/usr/bin/env python3
"""
FIXED Video Transformer Exercise Classifier

This is a corrected version that addresses common prediction issues:
1. Uses a simpler, more reliable architecture
2. Proper preprocessing for exercise videos
3. Better training strategy
4. Fallback to pre-trained features + simple classifier

Usage:
    python video_transformer_fixed.py
"""

print("🚀 Starting CPU-Optimized Video Transformer...")

# Import all required libraries
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pickle
from collections import deque, Counter
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("✅ All imports loaded successfully!")

print("=" * 70)
print("CPU-OPTIMIZED VIDEO TRANSFORMER EXERCISE CLASSIFIER")
print("=" * 70)

# CPU-Optimized Configuration for Maximum Accuracy
CONFIG = {
    'IMG_SIZE': 224,
    'NUM_FRAMES': 12,     # Reduced frames but better coverage
    'BATCH_SIZE': 4,      # Keep manageable batch size
    'LEARNING_RATE': 0.0005,  # Lower learning rate for CPU training
    'LR_REDUCE': 0.00005,  # Even lower reduced learning rate
    'EPOCHS': 80,         # More epochs for gradual convergence
    'TEMPORAL_WINDOW': 7, # Longer smoothing for stability
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'FEATURE_DIM': 512,
    'VALIDATION_SPLIT': 0.2,  # More validation data
    'DROPOUT': 0.5,       # Higher dropout to prevent overfitting
    'WEIGHT_DECAY': 0.001, # Stronger regularization
    'EARLY_STOPPING_PATIENCE': 10,  # More patience for CPU training
    'UNFREEZE_EPOCH': 15, # Earlier unfreezing
    'USE_MIXED_PRECISION': torch.cuda.is_available(),
    'CLASS_WEIGHTS': True, # Use class weights for balance
    'GRADIENT_ACCUMULATION_STEPS': 4,  # Simulate larger batches
}

print(f"✓ Device: {CONFIG['DEVICE']}")
print(f"✓ Config: {CONFIG['IMG_SIZE']}x{CONFIG['IMG_SIZE']}, {CONFIG['NUM_FRAMES']} frames/video (CPU-optimized)")
print(f"✓ Gradient accumulation steps: {CONFIG['GRADIENT_ACCUMULATION_STEPS']} (simulates batch size {CONFIG['BATCH_SIZE'] * CONFIG['GRADIENT_ACCUMULATION_STEPS']})")

class EnhancedVideoClassifier(nn.Module):
    """CPU-optimized video classifier for maximum accuracy with limited compute"""
    
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Use more powerful ResNet50 for better features
        print("🔄 Loading ResNet50 with CPU optimizations (this may take a moment)...")
        backbone_features = 2048  # Default for ResNet50
        try:
            self.backbone = models.resnet50(pretrained=True)
            backbone_features = 2048
            print("✅ ResNet50 loaded successfully")
        except Exception as e:
            print(f"⚠️ ResNet50 loading failed: {e}")
            print("🔄 Falling back to ResNet18...")
            try:
                self.backbone = models.resnet18(pretrained=True)
                backbone_features = 512  # ResNet18 output size
                print("✅ ResNet18 loaded successfully")
            except Exception as e2:
                print(f"⚠️ ResNet18 also failed: {e2}")
                print("🔄 Using ResNet18 without pretrained weights...")
                self.backbone = models.resnet18(pretrained=False)
                backbone_features = 512  # ResNet18 output size
                print("✅ ResNet18 (no pretrained) loaded successfully")
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Freeze most layers, keep only last 20 parameters trainable for CPU efficiency
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # CPU-optimized bidirectional GRU (good balance of accuracy and speed)
        self.gru = nn.GRU(
            input_size=backbone_features,  # Adaptive based on backbone
            hidden_size=256,  # Good hidden size for CPU
            num_layers=2,     # Reduced layers for CPU efficiency
            batch_first=True,
            dropout=CONFIG['DROPOUT'],
            bidirectional=True  # Bidirectional for better temporal understanding
        )
        
        # CPU-optimized attention mechanism with fewer heads
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # 256 * 2 (bidirectional)
            num_heads=4,    # Reduced heads for CPU efficiency
            dropout=CONFIG['DROPOUT']
        )
        
        # Enhanced classification head with more capacity
        self.classifier = nn.Sequential(
            nn.Dropout(CONFIG['DROPOUT']),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # LayerNorm instead of BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(CONFIG['DROPOUT']),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # LayerNorm instead of BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(CONFIG['DROPOUT']),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames = x.shape[:2]
        
        # Reshape to process all frames at once
        x = x.view(-1, *x.shape[2:])  # (batch_size * num_frames, C, H, W)
        
        # Extract features with backbone
        features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # Reshape back to sequence
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 2048)
        
        # Bidirectional GRU processing
        gru_out, _ = self.gru(features)  # (batch_size, num_frames, 512)
        
        # Apply attention mechanism
        # Transpose for attention: (seq_len, batch_size, embed_dim)
        gru_out_t = gru_out.transpose(0, 1)  # (num_frames, batch_size, 512)
        
        # Self-attention to focus on important frames
        attended_out, _ = self.attention(gru_out_t, gru_out_t, gru_out_t)
        
        # Global average pooling over time dimension
        attended_out = attended_out.transpose(0, 1)  # (batch_size, num_frames, 512)
        final_features = attended_out.mean(dim=1)  # (batch_size, 512)
        
        # Classification
        output = self.classifier(final_features)
        
        return output

class ExerciseVideoDataset(Dataset):
    """Enhanced dataset class for maximum precision"""
    
    def __init__(self, video_paths, labels, num_frames=8, img_size=224, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_training = is_training
        
        # Balanced training transforms - less aggressive to preserve exercise distinctions
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.2),  # Reduced horizontal flip
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Subtle transforms
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Gentle color changes
            transforms.RandomRotation(degrees=3),  # Minimal rotation to preserve pose
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Validation transform (no augmentation for consistent evaluation)
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        if frames is None:
            # Create dummy frames if loading fails
            frames = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
        
        return frames, torch.tensor(label, dtype=torch.long)
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return None
            
            # Sample frame indices
            if total_frames >= self.num_frames:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # Repeat frames if video is too short
                frame_indices = np.arange(total_frames)
                while len(frame_indices) < self.num_frames:
                    frame_indices = np.concatenate([frame_indices, frame_indices])
                frame_indices = frame_indices[:self.num_frames]
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Apply appropriate transforms based on training/validation
                    if self.is_training:
                        frame_tensor = self.train_transform(pil_image)
                    else:
                        frame_tensor = self.val_transform(pil_image)
                    frames.append(frame_tensor)
                else:
                    # Use last frame if reading fails
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # Create black frame
                        black_frame = torch.zeros(3, self.img_size, self.img_size)
                        frames.append(black_frame)
            
            cap.release()
            
            # Stack frames
            video_tensor = torch.stack(frames)  # (num_frames, 3, H, W)
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None

class FixedVideoTransformer:
    """Fixed real-time video transformer"""
    
    def __init__(self):
        self.device = torch.device(CONFIG['DEVICE'])
        self.model = None
        self.label_encoder = None
        self.frame_buffer = deque(maxlen=CONFIG['NUM_FRAMES'])
        self.prediction_history = deque(maxlen=CONFIG['TEMPORAL_WINDOW'])
        
        # Exercise classes - matching your actual folder names
        self.exercise_classes = [
            'barbell biceps curl',
            'hammer curl', 
            'push-up',
            'shoulder press',
            'squat'
        ]
        
        # Create mapping for display names (with underscores)
        self.display_names = [
            'barbell_biceps_curl',
            'hammer_curl', 
            'push_up',
            'shoulder_press',
            'squat'
        ]
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self, model_path="fixed_video_model.pth"):
        """Load trained model"""
        try:
            if os.path.exists(model_path):
                print(f"📚 Loading model from {model_path}")
                
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Initialize enhanced model
                self.model = EnhancedVideoClassifier(num_classes=len(self.exercise_classes))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                # Load label encoder
                self.label_encoder = checkpoint.get('label_encoder')
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                    self.label_encoder.fit(self.exercise_classes)
                
                print("✅ Model loaded successfully!")
                print(f"📋 Classes: {list(self.label_encoder.classes_)}")
                return True
            else:
                print(f"❌ Model file not found: {model_path}")
                print("💡 Training a new model...")
                return self.train_model()
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def add_frame(self, frame):
        """Add frame to buffer"""
        try:
            # Resize frame
            frame_resized = cv2.resize(frame, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            self.frame_buffer.append(pil_image)
        except Exception as e:
            print(f"Error adding frame: {e}")
    
    def predict(self):
        """Make prediction from current frame buffer"""
        if len(self.frame_buffer) < CONFIG['NUM_FRAMES'] or self.model is None:
            return None, 0.0
        
        try:
            # Prepare frames
            frames = list(self.frame_buffer)
            
            # Apply transforms
            frame_tensors = []
            for frame in frames:
                frame_tensor = self.transform(frame)
                frame_tensors.append(frame_tensor)
            
            # Stack frames
            video_tensor = torch.stack(frame_tensors)  # (num_frames, 3, H, W)
            video_tensor = video_tensor.unsqueeze(0).to(self.device)  # (1, num_frames, 3, H, W)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(video_tensor)
                probabilities = F.softmax(outputs, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Convert to exercise name and get display name
            folder_name = self.label_encoder.inverse_transform([predicted_class])[0]
            # Map to display name with underscores
            try:
                display_idx = self.exercise_classes.index(folder_name)
                exercise_name = self.display_names[display_idx]
            except ValueError:
                exercise_name = folder_name  # Fallback to original name
            
            # Add to prediction history
            self.prediction_history.append(exercise_name)
            
            return exercise_name, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def get_stable_prediction(self):
        """Get temporally smoothed prediction with confidence threshold"""
        if len(self.prediction_history) < 3:
            return "Detecting...", 0.0
        
        # Get most common prediction
        prediction_counts = Counter(self.prediction_history)
        most_common = prediction_counts.most_common(1)[0]
        prediction = most_common[0]
        stability = most_common[1] / len(self.prediction_history)
        
        # Only return confident predictions
        if stability < 0.6:  # Require 60% stability
            return "Detecting...", stability
        
        return prediction, stability
    
    def train_model(self):
        """Train the model"""
        print("🚀 Starting model training...")
        
        # Find dataset
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        dataset_locations = [
            os.path.join(PROJECT_ROOT, 'backend', 'dataset'),
            os.path.join(PROJECT_ROOT, 'dataset'),
            os.path.join(PROJECT_ROOT, 'data'),
            'dataset',
            'data'
        ]
        
        VIDEO_DIR = None
        for location in dataset_locations:
            if os.path.exists(location) and os.path.isdir(location):
                subdirs = [d for d in Path(location).iterdir() if d.is_dir()]
                if subdirs:
                    VIDEO_DIR = location
                    break
        
        if VIDEO_DIR is None:
            print("❌ No dataset found.")
            return False
        
        print(f"📁 Using dataset: {VIDEO_DIR}")
        
        # Collect ALL video files from ALL 5 exercise folders for maximum precision
        video_paths = []
        labels = []
        
        print("📁 Loading ALL videos from each exercise folder:")
        
        for class_dir in Path(VIDEO_DIR).iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name in self.exercise_classes:
                    class_idx = self.exercise_classes.index(class_name)
                    
                    # Get ALL video files (including .mp4, .avi, .mov, etc.)
                    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
                    class_videos = []
                    
                    for ext in video_extensions:
                        class_videos.extend(list(class_dir.glob(ext)))
                    
                    print(f"   📊 {class_name}: Found {len(class_videos)} videos")
                    
                    # Add ALL videos for this class
                    for video_file in class_videos:
                        video_paths.append(str(video_file))
                        labels.append(class_idx)
        
        if not video_paths:
            print("❌ No video files found")
            return False
        
        print(f"📊 Found {len(video_paths)} videos across {len(set(labels))} classes")
        
        # Setup label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.exercise_classes)
        
        # Split dataset - use smaller validation split to maximize training data
        X_train, X_val, y_train, y_val = train_test_split(
            video_paths, labels, 
            test_size=CONFIG['VALIDATION_SPLIT'], 
            random_state=42, 
            stratify=labels
        )
        
        print(f"📊 Training set: {len(X_train)} videos")
        print(f"📊 Validation set: {len(X_val)} videos")
        
        # Print class distribution
        from collections import Counter
        train_dist = Counter(y_train)
        val_dist = Counter(y_val)
        
        print("📊 Training distribution:")
        for class_idx, count in train_dist.items():
            class_name = self.exercise_classes[class_idx]
            print(f"   {class_name}: {count} videos")
        
        print("📊 Validation distribution:")
        for class_idx, count in val_dist.items():
            class_name = self.exercise_classes[class_idx]
            print(f"   {class_name}: {count} videos")
        
        # Create datasets with appropriate transforms
        train_dataset = ExerciseVideoDataset(X_train, y_train, CONFIG['NUM_FRAMES'], CONFIG['IMG_SIZE'], is_training=True)
        val_dataset = ExerciseVideoDataset(X_val, y_val, CONFIG['NUM_FRAMES'], CONFIG['IMG_SIZE'], is_training=False)
        
        # Create data loaders with drop_last to avoid batch size 1
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0, drop_last=False)
        
        # Initialize enhanced model
        self.model = EnhancedVideoClassifier(num_classes=len(self.exercise_classes)).to(self.device)
        
        # Setup enhanced optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=CONFIG['LEARNING_RATE'],
            weight_decay=CONFIG['WEIGHT_DECAY']
        )
        
        # Calculate class weights for balanced training
        if CONFIG['CLASS_WEIGHTS']:
            class_counts = Counter(y_train)
            total_samples = len(y_train)
            
            # Calculate inverse frequency weights
            class_weights = []
            for i in range(len(self.exercise_classes)):
                if i in class_counts:
                    weight = total_samples / (len(self.exercise_classes) * class_counts[i])
                    class_weights.append(weight)
                else:
                    class_weights.append(1.0)
            
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            print(f"📊 Class weights: {class_weights}")
            
            # Use weighted loss for class imbalance (no label smoothing for CPU training)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # CPU-optimized learning rate scheduling with more patience
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=CONFIG['LR_REDUCE']
        )
        
        # Mixed precision training for faster, cleaner training
        scaler = torch.cuda.amp.GradScaler() if CONFIG['USE_MIXED_PRECISION'] else None
        
        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(CONFIG['EPOCHS']):
            print(f"\n📈 Epoch {epoch+1}/{CONFIG['EPOCHS']}")
            
            # Unfreeze more layers after specified epoch for fine-tuning
            if epoch == CONFIG['UNFREEZE_EPOCH']:
                print(f"🔓 Unfreezing layer3 for fine-tuning...")
                for param in self.model.backbone.layer3.parameters():
                    param.requires_grad = True
            
            # Reduce learning rate after epoch 15
            if epoch == 15:
                print(f"📉 Reducing learning rate to {CONFIG['LR_REDUCE']}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = CONFIG['LR_REDUCE']
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Gradient accumulation for CPU training
            accumulation_steps = CONFIG['GRADIENT_ACCUMULATION_STEPS']
            optimizer.zero_grad()  # Initialize gradients
            
            for batch_idx, (videos, labels_batch) in enumerate(train_loader):
                videos = videos.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                # Mixed precision training
                if CONFIG['USE_MIXED_PRECISION'] and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(videos)
                        loss = criterion(outputs, labels_batch) / accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = self.model(videos)
                    loss = criterion(outputs, labels_batch) / accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels_batch.size(0)
                train_correct += (predicted == labels_batch).sum().item()
                
                if batch_idx % 5 == 0:
                    print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for videos, labels_batch in val_loader:
                    try:
                        videos = videos.to(self.device)
                        labels_batch = labels_batch.to(self.device)
                        
                        # Skip batch if size is 1 to avoid BatchNorm issues
                        if videos.size(0) == 1:
                            print(f"   ⚠️ Skipping validation batch with size 1")
                            continue
                        
                        outputs = self.model(videos)
                        loss = criterion(outputs, labels_batch)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels_batch.size(0)
                        val_correct += (predicted == labels_batch).sum().item()
                        
                    except Exception as e:
                        print(f"   ⚠️ Validation batch error: {e}")
                        continue
            
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
            
            print(f"   Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss/max(1, len(val_loader)):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Step scheduler with validation accuracy
            scheduler.step(val_acc)
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'label_encoder': self.label_encoder,
                    'config': CONFIG,
                    'val_acc': val_acc
                }, 'fixed_video_model.pth', _use_new_zipfile_serialization=False)
                print(f"   ✅ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement. Patience: {patience_counter}/{CONFIG['EARLY_STOPPING_PATIENCE']}")
                
                if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                    print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Generate confusion matrix for final evaluation
        print(f"\n🔬 Generating confusion matrix...")
        self.evaluate_model_confusion(val_loader)
        
        print(f"\n🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return True
    
    def evaluate_model_confusion(self, val_loader):
        """Generate confusion matrix to see which exercises are confused"""
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            
            self.model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for videos, labels_batch in val_loader:
                    videos = videos.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    outputs = self.model(videos)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels_batch.cpu().numpy())
            
            # Generate confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            
            # Create display names
            display_labels = [name.replace('_', ' ').title() for name in self.display_names]
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            plt.figure(figsize=(10, 8))
            disp.plot(cmap='Blues', values_format='d')
            plt.title('Exercise Classification Confusion Matrix')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Confusion matrix saved as 'confusion_matrix.png'")
            
            # Print which exercises are most confused
            print(f"\n🔍 Exercise Confusion Analysis:")
            for i, exercise in enumerate(display_labels):
                total = cm[i].sum()
                correct = cm[i, i]
                if total > 0:
                    accuracy = correct / total * 100
                    print(f"   {exercise}: {accuracy:.1f}% accuracy ({correct}/{total})")
                    
                    # Find most confused with
                    confused_idx = np.argmax(cm[i])
                    if confused_idx != i and cm[i, confused_idx] > 0:
                        confused_with = display_labels[confused_idx]
                        confusion_count = cm[i, confused_idx]
                        print(f"      Most confused with: {confused_with} ({confusion_count} times)")
            
        except ImportError:
            print("⚠️ Matplotlib not available. Skipping confusion matrix visualization.")
        except Exception as e:
            print(f"⚠️ Error generating confusion matrix: {e}")

def run_realtime_classification():
    """Run real-time exercise classification"""
    print("\n🎥 Starting real-time exercise classification...")
    
    # Initialize classifier
    classifier = FixedVideoTransformer()
    
    # Load model
    if not classifier.load_model():
        print("❌ Failed to load or train model")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized")
    print("📋 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 'r' to reset prediction history")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        fps = frame_count / (current_time - start_time)
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Add frame to classifier buffer
        classifier.add_frame(frame)
        
        # Make prediction every 10th frame
        if frame_count % 10 == 0:
            exercise, confidence = classifier.predict()
            if exercise:
                print(f"🏅 Detected: {exercise} (confidence: {confidence:.2f})")
        
        # Get stable prediction
        stable_exercise, stable_confidence = classifier.get_stable_prediction()
        
        # Draw overlay
        height, width = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw text
        exercise_text = f"Exercise: {stable_exercise.upper().replace('_', ' ')}"
        conf_text = f"Confidence: {stable_confidence:.1%}"
        fps_text = f"FPS: {fps:.1f} | Buffer: {len(classifier.frame_buffer)}/{CONFIG['NUM_FRAMES']}"
        
        # Color based on confidence
        if stable_confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif stable_confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, exercise_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, conf_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, fps_text, (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Fixed Video Transformer - Exercise Classifier', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            classifier.prediction_history.clear()
            classifier.frame_buffer.clear()
            print("🔄 Reset complete")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done")

if __name__ == "__main__":
    run_realtime_classification()
