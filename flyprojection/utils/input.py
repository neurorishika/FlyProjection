def get_boolean_answer(prompt, default=None):
    """Get a boolean answer from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt).lower()
        if answer == '':
            return default
        elif answer in ['y', 'yes']:
            return True
        elif answer in ['n', 'no']:
            return False
        else:
            print("Invalid answer. Please try again.")
            continue

def get_predefined_answer(prompt, options, default=None):
    """Get a predefined answer from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt).lower()
        if answer == '':
            return default
        elif answer in options:
            return answer
        else:
            print("Invalid answer. Please try again.")
            continue

def get_integer_answer(prompt, default=None):
    """Get an integer answer from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt)
        if answer == '':
            return default
        try:
            return int(answer)
        except ValueError:
            print("Invalid answer. Please try again.")
            continue

def get_float_answer(prompt, default=None):
    """Get a float answer from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt)
        if answer == '':
            return default
        try:
            return float(answer)
        except ValueError:
            print("Invalid answer. Please try again.")
            continue

def get_string_answer(prompt, default=None):
    """Get a string answer from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt)
        if answer == '':
            return default
        else:
            return answer

def get_file_path(prompt, default=None):
    """Get a file path from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt)
        if answer == '':
            return default
        elif os.path.exists(answer):
            return answer
        else:
            print("Invalid file path. Please try again.")
            continue

def get_directory_path(prompt, default=None):
    """Get a directory path from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt)
        if answer == '':
            return default
        elif os.path.isdir(answer):
            return answer
        else:
            print("Invalid directory path. Please try again.")
            continue

def get_color(prompt, default=None):
    """Get a color from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt)
        if answer == '':
            return default
        try:
            return hex_to_rgb(answer)
        except ValueError:
            print("Invalid color. Please try again.")
            continue

def get_date(prompt, default=None):
    """Get a date from the user. Defaults if no answer is given."""
    while True:
        answer = input(prompt + " (MM/DD/YYYY): ")
        if answer == '':
            return default
        try:
            return datetime.strptime(answer, "%m/%d/%Y")
        except ValueError:
            print("Invalid date. Please try again.")
            continue

# PySide2 based input functions
import sys
from PySide2.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox

def get_directory_path_qt(prompt, default=None):
    """Get a directory path from the user using a Qt dialog. Defaults if no answer is given."""
    app = QApplication.instance()  # Check if an instance already exists
    if app is None:
        app = QApplication(sys.argv)  # Create a new QApplication instance
    
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.ShowDirsOnly)
    if default:
        dialog.setDirectory(default)
    dialog.setWindowTitle(prompt)
    if dialog.exec_():
        return dialog.selectedFiles()[0]
    else:
        return default

def get_file_path_qt(prompt, default=None):
    """Get a file path from the user using a Qt dialog. Defaults if no answer is given."""
    app = QApplication.instance()  # Check if an instance already exists
    if app is None:
        app = QApplication(sys.argv)  # Create a new QApplication instance
    
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.ExistingFile)
    if default:
        dialog.setDirectory(default)
    dialog.setWindowTitle(prompt)
    if dialog.exec_():
        return dialog.selectedFiles()[0]
    else:
        return default

def get_string_answer_qt(prompt, default=None):
    """Get a string answer from the user using a Qt dialog. Defaults if no answer is given."""
    app = QApplication.instance()  # Check if an instance already exists
    if app is None:
        app = QApplication(sys.argv)  # Create a new QApplication instance

    answer, ok = QInputDialog.getText(None, prompt, prompt, text=default)
    if ok:
        return answer
    else:
        return default

def get_integer_answer_qt(prompt, default=None):
    """Get an integer answer from the user using a Qt dialog. Defaults if no answer is given."""
    app = QApplication.instance()  # Check if an instance already exists
    if app is None:
        app = QApplication(sys.argv)  # Create a new QApplication instance

    answer, ok = QInputDialog.getInt(None, prompt, prompt, value=default)
    if ok:
        return answer
    else:
        return default

def get_boolean_answer_qt(prompt):
    """Get a button press from the user using a Qt dialog."""
    app = QApplication.instance()  # Check if an instance already exists
    if app is None:
        app = QApplication(sys.argv)  # Create a new QApplication instance

    button = QMessageBox.question(None, prompt, prompt, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return button == QMessageBox.Yes
