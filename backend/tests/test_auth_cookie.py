from http.cookies import SimpleCookie
from types import SimpleNamespace

from fastapi import HTTPException, Response

from app.config import ACCESS_TOKEN_EXPIRE_MINUTES, AUTH_COOKIE_NAME
from app.dependencies import get_current_user
from app.models.user import User
from app.routers.auth import login, logout
from app.services.auth_service import create_access_token, hash_password


def add_user(db_session) -> User:
    user = User(
        username="cookieuser",
        email="cookie@example.com",
        password_hash=hash_password("correct-password"),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def get_set_cookie_header(response: Response) -> str:
    return response.headers["set-cookie"]


def get_auth_cookie_value(response: Response) -> str:
    cookie = SimpleCookie()
    cookie.load(get_set_cookie_header(response))
    return cookie[AUTH_COOKIE_NAME].value


def test_login_sets_httponly_cookie_and_me_reads_cookie(db_session) -> None:
    add_user(db_session)

    response = Response()
    login_response = login(
        response=response,
        form_data=SimpleNamespace(
            username="cookie@example.com",
            password="correct-password",
        ),
        db=db_session,
    )

    set_cookie = get_set_cookie_header(response)
    assert login_response == {"message": "Logged in"}
    assert AUTH_COOKIE_NAME in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Secure" not in set_cookie
    assert f"Max-Age={ACCESS_TOKEN_EXPIRE_MINUTES * 60}" in set_cookie

    current_user = get_current_user(
        request=SimpleNamespace(cookies={AUTH_COOKIE_NAME: get_auth_cookie_value(response)}),
        bearer_token=None,
        db=db_session,
    )

    assert current_user.email == "cookie@example.com"


def test_logout_clears_auth_cookie(db_session) -> None:
    response = Response()

    result = logout(response)

    assert result == {"message": "Logged out"}
    assert "Max-Age=0" in get_set_cookie_header(response)

    try:
        get_current_user(
            request=SimpleNamespace(cookies={}),
            bearer_token=None,
            db=db_session,
        )
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Expected missing credentials to be rejected")


def test_bearer_token_auth_still_works_for_backward_compatibility(
    db_session,
) -> None:
    user = add_user(db_session)
    token = create_access_token({"sub": str(user.id)})

    current_user = get_current_user(
        request=SimpleNamespace(cookies={}),
        bearer_token=token,
        db=db_session,
    )

    assert current_user.email == "cookie@example.com"
