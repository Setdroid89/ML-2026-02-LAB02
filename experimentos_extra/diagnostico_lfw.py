import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.datos.lfw import CargadorLFW
from src.datos.division import DivisorEstratificado
from src.embeddings.fabrica import FabricaExtractores
from src.clasificadores.fabrica import FabricaClasificadores

c = CargadorLFW().cargar(); p = DivisorEstratificado().dividir(c); C = c.n_clases
fe, fc = FabricaExtractores(), FabricaClasificadores()
d = 30
top5 = np.argsort(-np.bincount(c.y), kind="stable")[:5]
resto = np.setdiff1d(np.arange(C), top5)
preds = {}
print("1. Accuracy TRAIN / TEST (d=30)")
for emb in ("pca", "lda", "fisherfaces"):
    ext = fe.crear(emb, d, C).fit(p.X_train, p.y_train)
    Ztr, Zte = ext.transform(p.X_train), ext.transform(p.X_test)
    for mod in ("svm", "knn"):
        m = fc.crear(mod).fit(Ztr, p.y_train)
        ptr, pte = m.predict(Ztr), m.predict(Zte)
        preds[(emb, mod)] = pte
        print(f"  {emb:12} {mod:4} train={accuracy_score(p.y_train, ptr):.4f} test={accuracy_score(p.y_test, pte):.4f}")

bush = c.nombres_identidades.index("George W Bush")
print("\n2. Baseline")
print(f"  Bush id={bush}; proporción en TEST = {(p.y_test==bush).mean():.4f} ({(p.y_test==bush).sum()}/{len(p.y_test)})")
dm = DummyClassifier(strategy="most_frequent").fit(p.X_train, p.y_train)
pd_ = dm.predict(p.X_test)
print(f"  Dummy: accuracy={accuracy_score(p.y_test, pd_):.4f} macro-F1={f1_score(p.y_test, pd_, average='macro', zero_division=0):.4f}")

print("\n3-4. Recall por grupo y identidades con recall 0")
print("  top5:", [(c.nombres_identidades[i], int(np.sum(p.y_test==i))) for i in top5])
for emb, mod in (("lda","knn"), ("pca","svm"), ("fisherfaces","svm")):
    r = recall_score(p.y_test, preds[(emb, mod)], labels=np.arange(C), average=None, zero_division=0)
    print(f"  {emb}+{mod}: recall top5={np.round(r[top5],3).tolist()} (media {r[top5].mean():.4f}); "
          f"media otras 45={r[resto].mean():.4f}; identidades TEST con recall 0: {int((r==0).sum())}/{C}")
