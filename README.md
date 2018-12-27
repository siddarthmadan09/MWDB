# MWDB

Pre-Requisites
------------------------------------
1) python (version - 3x)
2) mongo db (running local on ports 27017)
3) pymongo python library
4) scipy python library
5) numpy python library
6) panda python libraary
7) tensorly python library
8) sklearn python library

Folder Structure
-----------------------------------------
README
Report
Outputs
Code
|_<task files>
|_db.py
|_data
  |_img
    |_<Visual descriptor files>
  |_<textual descriptor files>
  |_location xml file

Note: above folder structure have to be maintained for code to run properly

Execution
-------------------------------------------
1) Run mongo on local.
2) Run the db script present in Code folder using command python db.py 
3) run each task file using python <taskfile> <arg1> <arg2> 
Task 1 : ëpython3 task1.py <data space (user/image/location)> <decomposition method> <k-value>í
> python3 task1.py user svd 8

Task 2 : Execute task using command ëpython3 task1.py <data space (user/image/location)> <decomposition method> <k-value> <dataId>
> python3 task2.py image svd 8 2614112752

Task 3 : Execute task using command ëpython3 task3.py <Image ID> <Color Model><decomposition method><k-value>

Task 4 : Execute task using command ëpython3 task4.py <Location ID> <Color Model> <decomposition method> <k-value>

Task 5 : Execute task using command ëpython3 task5.py <Location ID> <k-value> <decomposition method>

Task 6 : Execute task using command ëpython3 task6.py <k-value> 

Task 7 : Execute task using command ëpython3 task7.py <k-value>
