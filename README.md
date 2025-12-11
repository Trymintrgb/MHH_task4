# MHH Task 4 – Deadlock Detection

This repository implements **deadlock detection by using ILP and BDD**. The goal is to report whether a deadlock exists in a given Petri net model — and if so, output one example deadlock marking. :contentReference[oaicite:1]{index=1}

---

## 📌 Overview

In this task:

- A deadlock is a reachable marking where no transition is enabled.
- A dead marking is any marking where no transition is enabled.
- We combine an **Integer Linear Programming (ILP)** formulation with the **Binary Decision Diagram (BDD)** from previous tasks to perform detection. :contentReference[oaicite:2]{index=2}

---

## 🧰 Requirements

Before running the code, set up a Python virtual environment:

```bash
python3 -m venv venv
Activate the virtual environment:

bash
Copy code
# Windows
venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
Then install requirements:

bash
Copy code
pip install -r requirements.txt
▶️ Running the Project
To run all tests with detailed output:

bash
Copy code
python3 -m pytest -vv test_Deadlock.py
To run a single specific test:

bash
Copy code
python3 -m pytest -vv test_Deadlock.py::test_001
🖼️ Result Example
Below is an example of the result image showing the detection output.
Replace assets/result.png with the path to your actual image file.

md
Copy code
![Deadlock Detection Result](assets/result.png)
👉 If your image is stored elsewhere, update the path or link accordingly.

📄 File Structure
Your project contains:

arduino
Copy code
├── src/
├── BTL_MHH.pdf
├── run.py
├── test_Deadlock.py
├── requirements.txt
├── README.md
📞 Contact
If you have questions or find issues, feel free to open an Issue on GitHub.

