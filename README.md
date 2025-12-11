# TASK 4 – BTL MHH

**Deadlock detection using ILP and BDD:**  
This task combines an ILP formulation with the BDD built in Task 3 to detect whether a deadlock exists.  
If a deadlock is reachable from the initial marking, return one concrete deadlock marking; otherwise, report that none exists.  
A *dead marking* is one where no transition is enabled; a *deadlock* is a dead marking reachable from the initial marking.  
You must also report running time for several example models.

---

## 🧰 Requirements

### 1. Tạo môi trường ảo (virtual environment)
```bash
python3 -m venv venv
2. Kích hoạt môi trường ảo
bash
Copy code
# Windows
venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
3. Cài đặt các thư viện từ requirements.txt
bash
Copy code
pip install -r requirements.txt
▶️ Running Tests
Run all tests
bash
Copy code
python3 -m pytest -vv test_Deadlock.py
Run a single test
bash
Copy code
python3 -m pytest -vv test_Deadlock.py::test_001
🖼️ Result Image (Insert Your Output Here)
Upload your result image into the repository (for example into a folder called assets/)
and place the line below after replacing the path with your actual image:

md
Copy code
![Deadlock Detection Result](assets/result.png)
Or use a raw GitHub link if you prefer.
