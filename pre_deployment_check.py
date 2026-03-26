"""
Complete pre-deployment checklist for SamiX.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def out(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="replace").decode("ascii"))


def print_section(title: str) -> None:
    out(f"\n{'=' * 70}")
    out(f"### {title}")
    out("=" * 70)


def run_checks() -> int:
    checks_passed = 0
    checks_failed = 0

    print_section("Python Environment")
    version = sys.version_info
    out(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 11:
        out("[PASS] Python 3.11+")
        checks_passed += 1
    else:
        out("[FAIL] Python 3.11+ required")
        checks_failed += 1

    print_section("Directory Structure")
    required_dirs = {
        "src": "Source code",
        "data": "Data directory",
        "data/api_responses": "API response storage",
        "data/auth": "User database",
        "data/backups": "Backups",
        "data/exports": "Exports",
        "data/kb": "Knowledge base",
        "data/history": "Audit history",
        ".streamlit": "Streamlit config",
        ".github": "CI/CD workflows",
    }
    for dir_path, desc in required_dirs.items():
        if Path(dir_path).exists():
            out(f"[PASS] {dir_path:25} {desc}")
            checks_passed += 1
        else:
            out(f"[FAIL] {dir_path:25} {desc}")
            checks_failed += 1

    print_section("Critical Files")
    required_files = {
        "app.py": "Streamlit entry point",
        "config.py": "Configuration management",
        "requirements.txt": "Python dependencies",
        ".streamlit/config.toml": "Streamlit settings",
        ".gitignore": "Git exclusions",
        "Dockerfile": "Docker configuration",
        "docker-compose.yml": "Docker Compose config",
        "Procfile": "Procfile",
        "validate.py": "Validation script",
        "quickstart.py": "Auto-setup script",
        "generate_hash.py": "Password hasher",
        "README.md": "Project guide",
    }
    for file_path, desc in required_files.items():
        if Path(file_path).exists():
            out(f"[PASS] {file_path:30} {desc}")
            checks_passed += 1
        else:
            out(f"[FAIL] {file_path:30} {desc}")
            checks_failed += 1

    print_section("Source Code Modules")
    modules = {
        ("src", "__init__.py"): "src package",
        ("src/auth", "authenticator.py"): "Authentication",
        ("src/db", "db_manager.py"): "SQLite manager",
        ("src/storage", "file_storage.py"): "API response storage",
        ("src/pipeline", "groq_client.py"): "Groq LLM client",
        ("src/pipeline", "stt_processor.py"): "STT processor",
        ("src/pipeline", "alert_engine.py"): "Alert system",
        ("src/ui", "login_page.py"): "Login UI",
        ("src/ui", "agent_panel.py"): "Agent panel",
        ("src/ui", "admin_panel.py"): "Admin panel",
        ("src/ui", "styles.py"): "Styling",
        ("src/utils", "kb_manager.py"): "Knowledge base",
        ("src/utils", "history_manager.py"): "History management",
        ("src/utils", "cost_tracker.py"): "Cost tracking",
        ("src/utils", "audio_processor.py"): "Audio utilities",
    }
    for (dir_path, file_name), desc in modules.items():
        full_path = Path(dir_path) / file_name
        if full_path.exists():
            out(f"[PASS] {str(full_path):40} {desc}")
            checks_passed += 1
        else:
            out(f"[FAIL] {str(full_path):40} {desc}")
            checks_failed += 1

    print_section("Python Packages")
    out("Checking installed packages...")
    packages = {
        "streamlit": ("streamlit", "Web framework"),
        "groq": ("groq", "Groq API client"),
        "deepgram-sdk": ("deepgram", "Deepgram STT"),
        "langchain": ("langchain", "LLM framework"),
        "pymilvus": ("pymilvus", "Vector database"),
        "bcrypt": ("bcrypt", "Password hashing"),
        "pydub": ("pydub", "Audio processing"),
        "python-dotenv": ("dotenv", "Environment variables"),
        "PyYAML": ("yaml", "YAML support"),
    }
    for package, (module_name, desc) in packages.items():
        try:
            __import__(module_name)
            out(f"[PASS] {package:25} {desc}")
            checks_passed += 1
        except ImportError:
            out(f"[FAIL] {package:25} {desc}")
            checks_failed += 1

    print_section("Configuration & Secrets")
    if Path(".env").exists():
        out("[PASS] .env file exists")
        checks_passed += 1
    else:
        out("[WARN] .env file missing")

    if Path(".streamlit/secrets.toml").exists():
        out("[PASS] .streamlit/secrets.toml exists")
        checks_passed += 1
    else:
        out("[WARN] .streamlit/secrets.toml missing")

    try:
        from config import Config

        groq_key = Config.get_groq_api_key()
        if groq_key and "gsk_" in groq_key:
            out("[PASS] GROQ_API_KEY configured")
            checks_passed += 1
        else:
            out("[WARN] GROQ_API_KEY not configured (template values)")
    except Exception as exc:
        out(f"[WARN] Could not verify API keys: {exc}")

    print_section("Import Test")
    imports_to_test = [
        ("config", "Config"),
        ("src.auth.authenticator", "AuthManager"),
        ("src.pipeline.groq_client", "GroqClient"),
        ("src.pipeline.stt_processor", "STTProcessor"),
        ("src.utils.kb_manager", "KBManager"),
        ("src.utils.history_manager", "HistoryManager"),
    ]
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            out(f"[PASS] {module_name:40} {class_name}")
            checks_passed += 1
        except Exception as exc:
            out(f"[FAIL] {module_name:40} {class_name}: {exc}")
            checks_failed += 1

    print_section("Git Configuration")
    try:
        result = subprocess.run(["git", "status"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            out("[PASS] Git repository initialized")
            checks_passed += 1
            gitignore_path = Path(".gitignore")
            if gitignore_path.exists():
                content = gitignore_path.read_text(encoding="utf-8")
                if ".env" in content and "secrets.toml" in content:
                    out("[PASS] .env and secrets.toml in .gitignore")
                    checks_passed += 1
                else:
                    out("[WARN] Sensitive files may not be excluded from git")
        else:
            out("[WARN] Git not initialized")
    except Exception as exc:
        out(f"[WARN] Git check failed: {exc}")

    print_section("Summary")
    total = checks_passed + checks_failed
    percentage = (checks_passed / total * 100) if total else 0
    out(f"Passed: {checks_passed}")
    out(f"Failed: {checks_failed}")
    out(f"Total:  {total}")
    out(f"Score:  {percentage:.1f}%\n")

    if checks_failed == 0:
        out("ALL CHECKS PASSED - READY FOR DEPLOYMENT\n")
        out("Next steps:")
        out("  1. Set environment variables or Streamlit secrets")
        out("  2. Run: streamlit run app.py")
        out("  3. Deploy to Streamlit Community Cloud")
        return 0

    out("[WARN] Some checks failed. Fix them before deployment.")
    return 1


if __name__ == "__main__":
    sys.exit(run_checks())
