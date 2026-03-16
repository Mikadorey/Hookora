from pydantic import BaseModel, EmailStr

# Used for registration
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


# Used for login
class UserLogin(BaseModel):
    username: str
    password: str


# Used when returning user data
class UserRead(BaseModel):
    id: int
    username: str
    email: EmailStr

    model_config = {
        "from_attributes": True
    }