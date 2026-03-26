"""
SamiX pre-flight validation script.
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def out(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


def check_python_version() -> bool:
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        out(f"[FAIL] Python {version.major}.{version.minor} detected")
        out("       Python 3.11+ is required")
        return False
    out(f"[PASS] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_directories() -> bool:
    required_dirs = [
        "data",
        "data/api_responses",
        "data/api_responses/transcriptions",
        "data/api_responses/llm_scores",
        "data/auth",
        "data/backups",
        "data/exports",
        "data/history",
        "data/kb",
        "data/uploads",
        "src",
        "src/auth",
        "src/db",
        "src/pipeline",
        "src/storage",
        "src/ui",
        "src/utils",
        ".streamlit",
    ]
    ok = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            out(f"[PASS] {dir_path}/")
        else:
            out(f"[FAIL] {dir_path}/ (missing)")
            ok = False
    return ok


def check_files() -> bool:
    required_files = [
        "app.py",
        "config.py",
        "requirements.txt",
        "src/db/schema.sql",
        "src/db/db_manager.py",
        "src/storage/file_storage.py",
        "src/auth/authenticator.py",
        "src/pipeline/groq_client.py",
        "src/pipeline/stt_processor.py",
        "src/pipeline/alert_engine.py",
        "src/ui/login_page.py",
        "src/ui/agent_panel.py",
        "src/ui/admin_panel.py",
        "src/ui/styles.py",
        "src/utils/kb_manager.py",
    ]
    ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            out(f"[PASS] {file_path}")
        else:
            out(f"[FAIL] {file_path} (missing)")
            ok = False
    return ok


def check_dependencies() -> bool:
    required_packages = {
        "streamlit": "streamlit",
        "groq": "groq",
        "streamlit_authenticator": "streamlit_authenticator",
        "langchain": "langchain",
        "pymilvus": "pymilvus",
        "bcrypt": "bcrypt",
        "pydub": "pydub",
        "dotenv": "dotenv",
        "yaml": "yaml",
    }
    out("\nChecking dependencies...")
    missing: list[str] = []
    for package, module_name in required_packages.items():
        try:
            import_module(module_name)
            out(f"[PASS] {package}")
        except ImportError:
            out(f"[FAIL] {package} (not installed)")
            missing.append(package)
    if missing:
        out(f"\n[WARN] Missing packages: {', '.join(missing)}")
        out(f"       Install with: pip install {' '.join(missing)}")
        return False
    return True


def check_environment_variables() -> bool:
    out("\nChecking environment variables...")
    try:
        from config import Config

        groq_key = Config.get_groq_api_key()
        if groq_key == "NOT_CONFIGURED" or "your_" in groq_key.lower():
            out("[FAIL] GROQ_API_KEY not configured")
            return False
        out(f"[PASS] GROQ_API_KEY configured (starts with: {groq_key[:8]}...)")

        deepgram_key = Config.get_deepgram_api_key()
        if "your_" in deepgram_key.lower():
            out("[INFO] DEEPGRAM_API_KEY not configured (local Whisper fallback will be used)")
        else:
            out(f"[PASS] DEEPGRAM_API_KEY configured (starts with: {deepgram_key[:8]}...)")
        return True
    except Exception as exc:
        out(f"[FAIL] Error loading config: {exc}")
        return False


def check_file_permissions() -> bool:
    out("\nChecking file permissions...")
    writable_dirs = ["data", "logs", ".streamlit"]
    ok = True
    for dir_path in writable_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
            out(f"[PASS] {dir_path}/ is writable")
        except Exception as exc:
            out(f"[FAIL] {dir_path}/ is not writable: {exc}")
            ok = False
    return ok


def main() -> int:
    print("\n" + "=" * 70)
    print("SamiX Pre-Flight Validation")
    print("=" * 70 + "\n")

    checks = [
        ("Python Version", check_python_version),
        ("Required Directories", check_directories),
        ("Required Files", check_files),
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_environment_variables),
        ("File Permissions", check_file_permissions),
    ]

    results: list[tuple[str, bool]] = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 70)
        try:
            results.append((check_name, check_func()))
        except Exception as exc:
            out(f"[FAIL] Error during validation: {exc}")
            results.append((check_name, False))

    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)
    for check_name, result in results:
        out(f"{'[PASS]' if result else '[FAIL]':8} {check_name}")

    out(f"\nResult: {passed}/{total} checks passed")
    if passed == total:
        out("\n[PASS] All checks passed. Ready to run:")
        out("       streamlit run app.py")
        return 0

    out("\n[WARN] Some checks failed. Review the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
