import datetime
from typing import Callable, Literal, Optional, cast
import sys, os


def check_module():
    try:
        import openai, azure.identity.broker  # type: ignore
    except ImportError:
        print("Please install the required packages by running the following command:")
        print("pip install openai azure-identity-broker --upgrade")
        exit(1)


check_module()


def get_openai_token(
    token_cache_file: str = "cloudgpt-apim-token-cache.bin",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    use_broker_login: Optional[bool] = None,
) -> str:
    """
    acquire token from Azure AD for CloudGPT OpenAI

    Parameters
    ----------
    token_cache_file : str, optional
        path to the token cache file, by default 'cloudgpt-apim-token-cache.bin' in the current directory
    client_id : Optional[str], optional
        client id for AAD app, by default None
    client_secret : Optional[str], optional
        client secret for AAD app, by default None

    Returns
    -------
    str
        access token for CloudGPT OpenAI
    """

    from azure.identity.broker import InteractiveBrowserBrokerCredential
    from azure.identity import (
        ManagedIdentityCredential,
        ClientSecretCredential,
        DeviceCodeCredential,
        AuthenticationRecord,
    )
    from azure.identity import TokenCachePersistenceOptions
    import msal

    api_scope_base = "api://feb7b661-cac7-44a8-8dc1-163b63c23df2"
    tenant_id = "72f988bf-86f1-41af-91ab-2d7cd011db47"
    scope = api_scope_base + "/.default"

    token_cache_option = TokenCachePersistenceOptions(
        name=token_cache_file,
        enable_persistence=True,
        allow_unencrypted_storage=True,
    )

    def save_auth_record(auth_record: AuthenticationRecord):
        try:
            with open(token_cache_file, "w") as cache_file:
                cache_file.write(auth_record.serialize())
        except Exception as e:
            print("failed to save auth record", e)

    def load_auth_record() -> Optional[AuthenticationRecord]:
        try:
            if not os.path.exists(token_cache_file):
                return None
            with open(token_cache_file, "r") as cache_file:
                return AuthenticationRecord.deserialize(cache_file.read())
        except Exception as e:
            print("failed to load auth record", e)
            return None

    auth_record: Optional[AuthenticationRecord] = load_auth_record()

    if client_id is not None:
        if client_secret is not None:
            identity = ClientSecretCredential(
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id,
                cache_persistence_options=token_cache_option,
                authentication_record=auth_record,
            )
        else:
            identity = ManagedIdentityCredential(
                client_id=client_id,
                cache_persistence_options=token_cache_option,
            )
    else:

        if use_broker_login is None:
            # enable broker login for known supported envs
            if sys.platform.startswith("darwin") or sys.platform.startswith("win32"):
                use_broker_login = True
            elif os.environ.get("WSL_DISTRO_NAME", "") != "":
                use_broker_login = True
            elif os.environ.get("TERM_PROGRAM", "") == "vscode":
                use_broker_login = True
            else:
                use_broker_login = False
        if use_broker_login:
            identity = InteractiveBrowserBrokerCredential(
                tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47",
                cache_persistence_options=token_cache_option,
                use_default_broker_account=True,
                parent_window_handle=msal.PublicClientApplication.CONSOLE_WINDOW_HANDLE,
                authentication_record=auth_record,
            )
        else:
            identity = DeviceCodeCredential(
                tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47",
                cache_persistence_options=token_cache_option,
                authentication_record=auth_record,
            )

    try:
        auth_record = cast(AuthenticationRecord, identity.authenticate(scopes=[scope]))  # type: ignore
        if auth_record:
            save_auth_record(auth_record)

    except Exception as e:
        print("failed to acquire token from AAD for CloudGPT OpenAI", e)
        raise e

    try:
        token = identity.get_token(scope)
        return token.token
    except Exception as e:
        print("failed to acquire token from AAD for CloudGPT OpenAI", e)
        raise e


cloudgpt_available_models = Literal[
    "gpt-35-turbo-20220309",
    "gpt-35-turbo-16k-20230613",
    "gpt-35-turbo-20230613",
    "gpt-35-turbo-1106",
    "gpt-4-20230321",
    "gpt-4-20230613",
    "gpt-4-32k-20230321",
    "gpt-4-32k-20230613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-visual-preview",
    "gpt-4-turbo-20240409",
    "gpt-4o-20240513",
    "gpt-4o-mini-20240718",
]


def encode_image(image_path: str, mime_type: Optional[str] = None) -> str:
    import base64
    import mimetypes

    file_name = os.path.basename(image_path)
    mime_type = (
        mime_type if mime_type is not None else mimetypes.guess_type(file_name)[0]
    )
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")

    if mime_type is None or not mime_type.startswith("image/"):
        print(
            "Warning: mime_type is not specified or not an image mime type. Defaulting to png."
        )
        mime_type = "image/png"

    image_url = f"data:{mime_type};base64," + encoded_image
    return image_url


def get_chat_completion(
    model: Optional[cloudgpt_available_models] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    *args,
    **kwargs,
):
    """
    helper function for getting chat completion from CloudGPT OpenAI
    """
    import openai

    engine = kwargs.get("engine")

    model_name = model
    if model_name is None:
        if engine is None:
            raise ValueError("model name must be specified by 'model' parameter")
        model_name = engine

    if "engine" in kwargs:
        del kwargs["engine"]

    client = openai.AzureOpenAI(
        api_version="2024-04-01-preview",
        azure_endpoint="https://cloudgpt-openai.azure-api.net/",
        azure_ad_token=get_openai_token(
            client_id=client_id, client_secret=client_secret
        ),
    )

    response = client.chat.completions.create(model=model_name, *args, **kwargs)

    return response


def auto_refresh_token(
    token_cache_file: str = "cloudgpt-apim-token-cache.bin",
    interval: datetime.timedelta = datetime.timedelta(minutes=15),
    on_token_update: Optional[Callable[[], None]] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Callable[[], None]:
    """
    helper function for auto refreshing token from CloudGPT OpenAI

    Parameters
    ----------
    token_cache_file : str, optional
        path to the token cache file, by default 'cloudgpt-apim-token-cache.bin' in the current directory
    interval : datetime.timedelta, optional
        interval for refreshing token, by default 15 minutes
    on_token_update : callable, optional
        callback function to be called when token is updated, by default None. In the callback function, you can get token from openai.api_key

    Returns
    -------
    callable
        a callable function that can be used to stop the auto refresh thread
    """

    import threading

    stop_signal = threading.Event()

    def update_token():
        import openai

        openai.api_type = "azure"
        openai.base_url = "https://cloudgpt-openai.azure-api.net/"
        openai.api_version = "2024-06-01"
        openai.api_key = get_openai_token(
            token_cache_file=token_cache_file,
            client_id=client_id,
            client_secret=client_secret,
        )

        if on_token_update is not None:
            on_token_update()

    def refresh_token_thread():
        while True:
            try:
                update_token()
            except Exception as e:
                print("failed to acquire token from AAD for CloudGPT OpenAI", e)

            if stop_signal.wait(interval.total_seconds()):
                break

    try:
        update_token()
    except Exception as e:
        raise Exception("failed to acquire token from AAD for CloudGPT OpenAI", e)

    thread = threading.Thread(target=refresh_token_thread, daemon=True)
    thread.start()

    def stop():
        stop_signal.set()
        try:
            thread.join()
        except Exception:
            pass

    return stop



def test_get_chat_completion():
    def test_call(*args, **kwargs):
        test_message = "What is the content?"
        test_chat_message = [{"role": "user", "content": test_message}]

        response = get_chat_completion(
            model="gpt-4o-mini-20240718",
            messages=test_chat_message,
            temperature=0.7,
            max_tokens=100,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            *args,
            **kwargs,
        )

        print(response.choices[0].message)

    print("test without AAD app")
    test_call()  # test without AAD app


if __name__ == "__main__":
    test_get_chat_completion()
