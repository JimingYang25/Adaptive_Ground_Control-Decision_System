cmake_minimum_required(VERSION 3.8)
project(terrain_classifier_pkg)

find_package(ament_cmake REQUIRED)

execute_process(COMMAND python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" OUTPUT_VARIABLE PYTHON_VERSION)
string(STRIP ${PYTHON_VERSION} PYTHON_VERSION)
install(DIRECTORY terrain_classifier_pkg/ DESTINATION lib/${PYTHON_VERSION}/site-packages/${PROJECT_NAME})

install(
  PROGRAMS
    scripts/terrain_node.py
  DESTINATION
    lib/${PROJECT_NAME}
)

# install Python Module
install(
  DIRECTORY
    terrain_classifier_pkg/
  DESTINATION
    lib/python3.12/site-packages/${PROJECT_NAME}  # 注意 Python 版本号
  PATTERN "__pycache__" EXCLUDE
)

# install launch file
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# install config file
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)


# install model files
install(
  DIRECTORY
    models/
  DESTINATION
    share/${PROJECT_NAME}/models
)

ament_package()