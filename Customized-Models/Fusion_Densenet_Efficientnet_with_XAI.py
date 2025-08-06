from multiprocessing import freeze_support
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.svm import LinearSVC
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries

def main():
    
    input_path = r"D:\Research\Topics\GastrointestinalBleeding\aug"
    best_model_file = r"D:\Research\Topics\GastrointestinalBleeding\output\model/best_model1.pth"

    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    IMG_SIZE = (224, 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    
     # For LIME batch predict
    lime_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load dataset and create splits
    full_dataset = datasets.ImageFolder(input_path, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    indices = torch.randperm(total)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(Subset(full_dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    
    class_to_val_idxs = {i: [] for i in range(len(class_names))}
    for idx in val_idx.tolist():
        _, label = full_dataset.samples[idx]
        class_to_val_idxs[label].append(idx)

    # Define fusion model
    class FusionModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # DenseNet121 backbone
            self.densenet = models.densenet121(pretrained=True)
            self.densenet.classifier = nn.Identity()
            for p in self.densenet.parameters(): p.requires_grad = False
            # EfficientNetB0 backbone
            self.efficientnet = models.efficientnet_b0(pretrained=True)
            self.efficientnet.classifier = nn.Identity()
            for p in self.efficientnet.parameters(): p.requires_grad = False
            # Fusion head
            fusion_dim = 1024 + 1280
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            f1 = self.densenet(x)
            f2 = self.efficientnet(x)
            return self.classifier(torch.cat([f1, f2], dim=1))

    
    model = FusionModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

   
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        model.train()
        running_loss, running_corrects = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * X.size(0)
            running_corrects += (preds == y).sum().item()
        train_loss = running_loss / train_size
        train_acc = running_corrects / train_size
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * X.size(0)
                val_corrects += (preds == y).sum().item()
        val_loss /= val_size
        val_acc = val_corrects / val_size
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_file)
            print(f"New best model saved with Val Acc: {best_acc:.4f}")
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")

    # Plot accuracy & loss curves
    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'],   label='Val Acc')
    plt.title('Accuracy over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.title('Loss over Epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.show()

    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_file))
    model.eval()

    # Evaluate on validation dataset
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.show()

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=class_names))

    y_scores = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            y_scores.extend(probs.cpu().numpy())
    y_scores = np.array(y_scores)[:, 1] 

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.title('Binary ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    

    # --- Grad‑CAM++ 
    target_layers = [ model.densenet.features.denseblock4.denselayer16.conv2 ]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    os.makedirs("gradcampp_outputs", exist_ok=True)

    # --- For each class, one Grad‑CAM++ plot ---
    for class_idx, cls_name in enumerate(class_names):
        idx = class_to_val_idxs[class_idx][0]  # pick first val sample
        img_path, _ = full_dataset.samples[idx]

        # read & preprocess for overlay
        orig_bgr = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = cv2.resize(orig_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
        orig_float = orig_rgb.astype(np.float32) / 255.0

        tensor, _ = full_dataset[idx]
        input_t = tensor.unsqueeze(0).to(device); input_t.requires_grad_()

        grayscale_cam = cam(input_t,
                            targets=[ClassifierOutputTarget(class_idx)],
                            aug_smooth=True,
                            eigen_smooth=True)[0]
        cam_image = show_cam_on_image(orig_float, grayscale_cam,
                                      use_rgb=True, image_weight=0.6)

        plt.figure(figsize=(6,4))
        plt.suptitle(f"Grad‑CAM++: {cls_name}", fontsize=14)
        plt.subplot(1,2,1); plt.imshow(orig_rgb);  plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(cam_image); plt.axis("off")
        plt.show()

    del cam
    print("All Grad‑CAM++ images saved in ./gradcampp_outputs/")

    # --- LIME ---
    explainer = lime_image.LimeImageExplainer()
    def batch_predict(images):
        model.eval()
        batch = torch.stack([lime_transform(img) for img in images], dim=0).to(device).float()
        logits = model(batch)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()


    for class_idx, cls_name in enumerate(class_names):
        idx = class_to_val_idxs[class_idx][0]
        img_path, _ = full_dataset.samples[idx]

        orig_bgr = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = cv2.resize(orig_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
        orig_float = orig_rgb.astype(np.double) / 255.0

        explanation = explainer.explain_instance(
            image=orig_float,
            classifier_fn=batch_predict,
            top_labels=2,
            hide_color=0,
            num_samples=1000,
            segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10)
        )
        temp, mask = explanation.get_image_and_mask(
            label=class_idx,
            positive_only=True,
            num_features=5,
            hide_rest=False
        )

        plt.figure(figsize=(4,4))
        plt.title(f"LIME: {cls_name}")
        plt.imshow(mark_boundaries(temp, mask))
        plt.axis("off")
        plt.show()


    # --- Relative TCAV across layers ---
    layers = {
        'block1': model.densenet.features.denseblock1.denselayer1.conv2,
        'block2': model.densenet.features.denseblock2.denselayer6.conv2,
        'block3': model.densenet.features.denseblock3.denselayer12.conv2,
        'block4': model.densenet.features.denseblock4.denselayer16.conv2,
        'fc':     model.classifier[0]
    }

    def compute_tcav_for_layer(layer_module):
        feats = {}
        # proper hook function
        def hook(module, inp, out):
            feats['acts'] = out
            if not out.requires_grad:
                out.requires_grad_(True)
            out.register_hook(lambda g: feats.setdefault('grads', g.clone().detach()))
        handle = layer_module.register_forward_hook(hook)

        C = len(class_names)
        scores = np.zeros((C, C))
        for c_idx in range(C):
            # gather concept activations
            concept_acts = []
            for idx in class_to_val_idxs[c_idx]:
                x, _ = full_dataset[idx]
                x = x.unsqueeze(0).to(device)
                model(x)
                act = feats['acts']
                if act.ndim == 4:
                    act = act.mean(dim=(2,3))
                concept_acts.append(act.detach().cpu().numpy().ravel())
            concept_acts = np.stack(concept_acts)

            # gather random activations
            other_idxs = [i for i in val_idx.tolist() if full_dataset.samples[i][1]!=c_idx]
            sampled = np.random.choice(other_idxs, min(len(other_idxs),100), replace=False)
            random_acts = []
            for idx in sampled:
                x, _ = full_dataset[idx]
                x = x.unsqueeze(0).to(device)
                model(x)
                act = feats['acts']
                if act.ndim == 4:
                    act = act.mean(dim=(2,3))
                random_acts.append(act.detach().cpu().numpy().ravel())
            random_acts = np.stack(random_acts)

            # train CAV
            X = np.vstack([concept_acts, random_acts])
            y = np.hstack([np.ones(len(concept_acts)), np.zeros(len(random_acts))])
            cav = LinearSVC().fit(X, y)

            # compute TCAV for each target
            for t_idx in range(C):
                pos = 0
                total = len(class_to_val_idxs[t_idx])
                for idx in class_to_val_idxs[t_idx]:
                    model.zero_grad()
                    x, _ = full_dataset[idx]
                    x = x.unsqueeze(0).to(device)
                    out = model(x)
                    out[0,t_idx].backward(retain_graph=True)
                    grad = feats['grads']
                    if grad.ndim == 4:
                        grad = grad.mean(dim=(2,3))
                    g = grad.cpu().numpy().ravel()
                    if np.dot(cav.coef_, g) > 0:
                        pos += 1
                scores[c_idx, t_idx] = pos/total if total else 0.0

        handle.remove()
        return scores

    layer_scores = {name: compute_tcav_for_layer(layer) for name, layer in layers.items()}

    # plot same-concept sensitivity
    layer_names   = list(layers.keys())
    lesion_scores = [layer_scores[l][0,0] for l in layer_names]
    normal_scores = [layer_scores[l][1,1] for l in layer_names]

    plt.figure(figsize=(8,5))
    plt.plot(layer_names, lesion_scores, '-o', label='Lesion→Lesion')
    plt.plot(layer_names, normal_scores, '-o', label='Normal→Normal')
    plt.xlabel('Layer')
    plt.ylabel('TCAV Score')
    plt.title('Relative TCAV (Same‑Concept) Across Layers')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot grouped bar chart with zero markers
    concepts    = class_names
    n_concepts  = len(concepts)
    layer_names = list(layers.keys())
    scores      = np.array([[layer_scores[l][i,i] for i in range(n_concepts)]
                            for l in layer_names])

    x = np.arange(n_concepts)
    width = 0.8 / len(layer_names)

    fig, ax = plt.subplots(figsize=(8,5))
    bar_containers = []
    for idx, layer in enumerate(layer_names):
        bc = ax.bar(x + idx*width, scores[idx], width, label=layer)
        bar_containers.append(bc)

    # annotate zero bars with '*'
    for idx, bc in enumerate(bar_containers):
        for j, bar in enumerate(bc):
            if scores[idx,j] == 0:
                x_pos = bar.get_x() + bar.get_width()/2
                ax.text(x_pos, -0.02, '*',
                        ha='center', va='top',
                        color=bar.get_facecolor(),
                        fontsize=14,
                        clip_on=False)

    ax.set_xlabel('Concept')
    ax.set_ylabel('TCAV Score')
    ax.set_title('TCAV (same-concept) per Layer')
    ax.set_xticks(x + width*(len(layer_names)-1)/2)
    ax.set_xticklabels(concepts)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(title='Layer', loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()


    print('All processes complete.')


if __name__ == "__main__":
    freeze_support()
    main()
