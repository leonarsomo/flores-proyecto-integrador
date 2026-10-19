"""Pruebas de los componentes del ajuste fino de la A2 (sin descargar pesos)."""
import numpy as np
import pytest
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flores.ajuste_fino import BACKBONE, ColaViT, Configuracion, LoRALineal, sonda_lineal, tokens_hasta


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    return timm.create_model(BACKBONE, pretrained=False, num_classes=0, img_size=224).eval()


def test_lora_arranca_identica_y_se_fusiona_sin_error():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 12)
    capa = LoRALineal(base, r=4, alfa=8)
    x = torch.randn(5, 16)
    assert torch.allclose(capa(x), base(x))                 # B = 0 al inicio: misma función
    with torch.no_grad():
        capa.B.normal_()
    assert torch.allclose(capa(x), capa.fusionar()(x), atol=1e-5)   # fusión exacta para desplegar
    assert not any(q.requires_grad for q in capa.base.parameters())


@pytest.mark.parametrize("kw,esperado", [
    (dict(ajustar=2), 2 * 1_775_232 + 768 + 1_925),
    (dict(ajustar=4), 4 * 1_775_232 + 768 + 1_925),
    (dict(ajustar=4, lora_r=8), 4 * 8 * ((384 + 1152) + (384 + 384)) + 1_925),
])
def test_parametros_entrenables(vit, kw, esperado):
    assert ColaViT(vit, 8, **kw).entrenables() == esperado


def test_cola_reproduce_el_vit_completo(vit):
    """La cola sobre la caché del bloque 8 da el mismo token de clase que el ViT entero."""
    x = torch.randn(2, 3, 224, 224)
    cola = ColaViT(vit, 8, 0).eval()
    with torch.no_grad():
        completo = vit.forward_features(x)[:, 0]
        por_cache = cola.representacion(tokens_hasta(vit, x, 8))
    assert torch.allclose(completo, por_cache, atol=1e-4)


def test_sonda_plegada_equivale_a_la_tuberia_de_E03(vit):
    """W' = W / s y b' = b - W' m reproducen StandardScaler + regresión logística."""
    torch.manual_seed(1)
    cola = ColaViT(vit, 8, 0).eval()
    tok = torch.randn(60, 257, 384) * 0.5
    y = torch.arange(60) % 5
    sonda_lineal(cola, tok, y, Configuracion(c_sonda=0.1), torch.Generator())
    with torch.no_grad():
        feats = cola.representacion(tok).numpy()
        z = cola(tok).numpy()
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(feats, y.numpy())
    assert np.allclose(z, ref.decision_function(feats), atol=1e-3)
