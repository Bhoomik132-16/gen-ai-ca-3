"""
sample_data.py
==============
Provides an extensive, rich mock dataset for QueryMate.
Includes:
- 50 Students across multiple Computer Science divisions (A, B, C)
- 6 Core Subjects: Generative AI, Data Structures, DBMS, Operating Systems,
  Computer Networks, and Machine Learning
- 900 Examination marks across Unit 1, Unit 2, Unit 3 (with high, average, and low scores)
- 300 Subject attendance records (classes held vs classes attended)
"""

import random

STUDENTS_DATA = [
    (1, "Aarav Sharma", "B.Tech CSE", "A"),
    (2, "Priya Patel", "B.Tech CSE", "A"),
    (3, "Rohan Verma", "B.Tech CSE", "A"),
    (4, "Ananya Iyer", "B.Tech CSE", "A"),
    (5, "Vikram Singh", "B.Tech CSE", "A"),
    (6, "Diya Nair", "B.Tech CSE", "A"),
    (7, "Aditya Kulkarni", "B.Tech CSE", "A"),
    (8, "Sneha Mukherjee", "B.Tech CSE", "A"),
    (9, "Kavya Reddy", "B.Tech CSE", "A"),
    (10, "Arjun Gupta", "B.Tech CSE", "A"),
    (11, "Meera Joshi", "B.Tech CSE", "A"),
    (12, "Siddharth Rao", "B.Tech CSE", "A"),
    (13, "Ishaan Malhotra", "B.Tech CSE", "A"),
    (14, "Tanvi Deshmukh", "B.Tech CSE", "A"),
    (15, "Varun Chawla", "B.Tech CSE", "A"),
    (16, "Pooja Hegde", "B.Tech CSE", "A"),
    (17, "Karan Mehta", "B.Tech CSE", "A"),
    (18, "Rhea Sengupta", "B.Tech CSE", "A"),
    (19, "Nikhil Pillai", "B.Tech CSE", "A"),
    (20, "Shreya Bhat", "B.Tech CSE", "A"),
    (21, "Devansh Mishra", "B.Tech CSE", "B"),
    (22, "Anushka Sen", "B.Tech CSE", "B"),
    (23, "Harsh Agarwal", "B.Tech CSE", "B"),
    (24, "Rashi Khanna", "B.Tech CSE", "B"),
    (25, "Manish Tiwari", "B.Tech CSE", "B"),
    (26, "Ritika Choudhary", "B.Tech CSE", "B"),
    (27, "Yashvardhan Jain", "B.Tech CSE", "B"),
    (28, "Bhavna Kapoor", "B.Tech CSE", "B"),
    (29, "Ayush Saxena", "B.Tech CSE", "B"),
    (30, "Nehal Chauhan", "B.Tech CSE", "B"),
    (31, "Pranav Nambiar", "B.Tech CSE", "B"),
    (32, "Kriti Sanon", "B.Tech CSE", "B"),
    (33, "Abhinav Das", "B.Tech CSE", "B"),
    (34, "Garima Tripathi", "B.Tech CSE", "B"),
    (35, "Tushar Rawat", "B.Tech CSE", "B"),
    (36, "Isha Ganguly", "B.Tech CSE", "C"),
    (37, "Omkar Patil", "B.Tech CSE", "C"),
    (38, "Pallavi Soni", "B.Tech CSE", "C"),
    (39, "Chirag Singhal", "B.Tech CSE", "C"),
    (40, "Aishwarya Roy", "B.Tech CSE", "C"),
    (41, "Kunal Goswami", "B.Tech CSE", "C"),
    (42, "Shruti Pandey", "B.Tech CSE", "C"),
    (43, "Sameer Qureshi", "B.Tech CSE", "C"),
    (44, "Nandini Ghosh", "B.Tech CSE", "C"),
    (45, "Gaurav Bhattacharya", "B.Tech CSE", "C"),
    (46, "Sanya Bansal", "B.Tech CSE", "C"),
    (47, "Mohit Kaushik", "B.Tech CSE", "C"),
    (48, "Deepika Menon", "B.Tech CSE", "C"),
    (49, "Tarun Mathur", "B.Tech CSE", "C"),
    (50, "Vidya Sundaram", "B.Tech CSE", "C"),
]

SUBJECTS_DATA = [
    (1, "Generative AI"),
    (2, "Data Structures"),
    (3, "DBMS"),
    (4, "Operating Systems"),
    (5, "Computer Networks"),
    (6, "Machine Learning"),
]


def generate_marks_data():
    """
    Generates 900 deterministic marks entries for 50 students across
    6 subjects and 3 units (Unit 1, Unit 2, Unit 3).
    Ensures realistic variety: toppers (>80), averages (50-75), and students (<40).
    """
    marks = []
    mark_id = 1
    units = ["Unit 1", "Unit 2", "Unit 3"]

    # Seeded pseudo-random generation for repeatable academic realism
    rng = random.Random(42)

    for student_id in range(1, 51):
        # Assign an academic tier to each student
        if student_id in [1, 2, 8, 11, 14, 20, 22, 27, 40, 50]:
            # Tier 1: High performers (scores 80 - 99)
            base_min, base_max = 82, 98
        elif student_id in [3, 5, 12, 17, 25, 35, 41, 47]:
            # Tier 3: Struggling / Remedial (scores 25 - 48, with <40)
            base_min, base_max = 28, 48
        else:
            # Tier 2: Consistent / Average (scores 50 - 79)
            base_min, base_max = 52, 79

        for subject_id in range(1, 7):
            for unit in units:
                score = rng.randint(base_min, base_max)
                # Cap between 10 and 100
                score = max(15, min(100, score))
                marks.append((mark_id, student_id, subject_id, unit, score, 100))
                mark_id += 1

    return marks


def generate_attendance_data():
    """
    Generates 300 attendance records for 50 students across 6 subjects.
    Classes held is between 38-42.
    Includes students with low attendance (<75%) and stellar attendance (>90%).
    """
    attendance = []
    rng = random.Random(99)

    for student_id in range(1, 51):
        if student_id in [3, 5, 10, 12, 17, 23, 29, 35, 41, 47]:
            # Low attendance profile (50% - 73%)
            rate_min, rate_max = 0.52, 0.73
        elif student_id in [1, 2, 8, 14, 20, 22, 32, 40, 50]:
            # High attendance profile (90% - 98%)
            rate_min, rate_max = 0.90, 0.98
        else:
            # Regular profile (75% - 89%)
            rate_min, rate_max = 0.75, 0.89

        for subject_id in range(1, 7):
            classes_held = rng.randint(38, 42)
            rate = rng.uniform(rate_min, rate_max)
            attended = int(round(classes_held * rate))
            attended = max(10, min(classes_held, attended))
            attendance.append((student_id, subject_id, classes_held, attended))

    return attendance
