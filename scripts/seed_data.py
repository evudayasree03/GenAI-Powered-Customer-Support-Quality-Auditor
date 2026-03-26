from _bootstrap import setup_project_root

setup_project_root()

from src.auth.authenticator import AuthManager


def main() -> None:
    auth = AuthManager()
    auth.register("agent@samix.ai", "Sample Agent", "agent123")
    auth.register("admin@samix.ai", "Admin", "admin")
    print("Seeded sample users.")


if __name__ == "__main__":
    main()
