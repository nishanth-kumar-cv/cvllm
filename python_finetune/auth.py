from fastapi import Header, HTTPException, status

ADMIN_TOKEN = "admin-secret-token"
USER_TOKEN = "user-secret-token"

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    token = authorization.split(" ")[1]
    if token not in [ADMIN_TOKEN, USER_TOKEN]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return token
