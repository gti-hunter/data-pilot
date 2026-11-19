import pandas as pd
import pytest


@pytest.fixture
def df() -> pd.DataFrame:
    """Base mínima com resultado conhecido (calculado à mão)."""
    linhas = [
        ("P1", "2026-01-10", "A", "Cidade1", "Cat1", 100.0, "concluído"),
        ("P2", "2026-01-20", "B", "Cidade2", "Cat1", 200.0, "concluído"),
        ("P3", "2026-02-05", "A", "Cidade1", "Cat2", 300.0, "concluído"),
        ("P4", "2026-02-15", "C", "Cidade2", "Cat2", 400.0, "cancelado"),
        ("P5", "2026-03-01", "D", "Cidade3", "Cat1", 500.0, "pendente"),
    ]
    d = pd.DataFrame(linhas, columns=["id_pedido", "data", "cliente", "cidade", "categoria", "receita", "status"])
    d["data"] = pd.to_datetime(d["data"])
    return d
