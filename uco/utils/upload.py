import subprocess


COMPETITION_NAME = "understanding_cloud_organization"


def kaggle_submit(csv_filename, submission_name):
    p = subprocess.Popen(
        [
            "kaggle",
            "competitions",
            "submit",
            "-f",
            str(csv_filename),
            "-m",
            submission_name,
            COMPETITION_NAME,
        ]
    )
    p.communicate()
