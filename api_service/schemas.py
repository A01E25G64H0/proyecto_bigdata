from pydantic import BaseModel, Field
from typing import Literal

class TransactionInput(BaseModel):
    step: int = Field(..., ge=1, le=744, description="Hora de simulación (1 a 744)")
    type: Literal["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    amount: float = Field(..., gt=0, le=10_000_000, description="Monto mayor a 0")
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)
    is_merchant_dest: int = Field(..., ge=0, le=1)
