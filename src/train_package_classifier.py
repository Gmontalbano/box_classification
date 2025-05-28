from pathlib import Path
import random, warnings, numpy as np, torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.datasets.folder import pil_loader as default_loader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore', category=UserWarning, message='Palette images')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('🖥️ Usando', 'GPU: '+torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU.')

BASE_DIR = Path('utils/dados')
DATA_DIR = BASE_DIR / 'damaged-and-intact-packages'
CKPT_DIR = BASE_DIR / 'models' / 'package_inspection'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_DIR.exists(), f"❌ {DATA_DIR} não encontrado."

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def rgb_loader(path): return Image.open(path).convert('RGB')

full_ds = datasets.ImageFolder(str(DATA_DIR), transform=transform, loader=rgb_loader)
labels = full_ds.targets
train_idx, tmp_idx = train_test_split(range(len(full_ds)), test_size=0.2, stratify=labels, random_state=SEED)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=[labels[i] for i in tmp_idx], random_state=SEED)

train_ds = Subset(full_ds, train_idx)
val_ds   = Subset(full_ds, val_idx)
test_ds  = Subset(full_ds, test_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
for n, p in model.features.named_parameters():
    p.requires_grad = int(n.split('.')[0]) >= 14
model.classifier[1] = nn.Linear(1280, 2)
model.to(device)

crit = nn.CrossEntropyLoss()
opt  = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.3, 2)

EPOCHS = 8
history, best = {'tr_loss': [], 'vl_loss': [], 'tr_acc': [], 'vl_acc': []}, 0

for ep in range(1, EPOCHS+1):
    model.train(); tl, tp, tt = 0, [], []
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); o = model(x); loss = crit(o,y); loss.backward(); opt.step()
        tl += loss.item()*len(x); tp += o.argmax(1).tolist(); tt += y.tolist()
    tr_loss = tl / len(train_ds); tr_acc = accuracy_score(tt, tp)

    model.eval(); vl, vp, vt = 0, [], []
    with torch.no_grad():
        for x,y in val_loader:
            o = model(x.to(device)); vl += crit(o,y.to(device)).item()*len(x)
            vp += o.argmax(1).cpu().tolist(); vt += y.tolist()
    vl_loss = vl / len(val_ds); vl_acc = accuracy_score(vt, vp); sched.step(vl_acc)

    history['tr_loss'].append(tr_loss); history['vl_loss'].append(vl_loss)
    history['tr_acc'].append(tr_acc); history['vl_acc'].append(vl_acc)
    print(f"Epoch {ep}: TL {tr_loss:.3f} TA {tr_acc*100:.1f}% | VL {vl_loss:.3f} VA {vl_acc*100:.1f}%")

    if vl_acc > best:
        best = vl_acc
        torch.save(model.state_dict(), CKPT_DIR / 'best.pth')
        print('  ↳ Melhor modelo salvo')

model.load_state_dict(torch.load(CKPT_DIR / 'best.pth', map_location=device))
model.eval(); pp, tt = [], []
with torch.no_grad():
    for x,y in test_loader:
        pp += model(x.to(device)).argmax(1).cpu().tolist()
        tt += y.tolist()

print(classification_report(tt, pp, target_names=full_ds.classes))
ConfusionMatrixDisplay(confusion_matrix(tt, pp), display_labels=full_ds.classes).plot(cmap='Blues')
plt.show()

plt.plot(history['tr_loss'], label='Train'); plt.plot(history['vl_loss'], label='Val')
plt.title('Loss'); plt.xlabel('Época'); plt.legend(); plt.show()

plt.plot(history['tr_acc'], label='Train'); plt.plot(history['vl_acc'], label='Val')
plt.title('Acurácia'); plt.xlabel('Época'); plt.legend(); plt.show()

# load_model.py
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 2)
model.load_state_dict(torch.load(CKPT_DIR / 'best.pth'))
model.to(device).eval()
