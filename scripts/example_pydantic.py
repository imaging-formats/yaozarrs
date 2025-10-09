from pydantic import BaseModel


class A(BaseModel):
    a: int
    b: str


class B(BaseModel):
    x: float
    y: list[A]
    z: dict[str, A]


x = B(x=1.0, y=[{"a": 1, "b": ["hello"]}, {"a": "2f3", "b": "world"}], z="value")
print(x)
