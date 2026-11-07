"""Compile and execute code, capturing output and runtime."""
import os
import shlex
import subprocess
import time

from .format_colors import color


def compile_code(source_file, lang, code_compiler, code_compiler_kwds, include_dir=None):
       
    # Split the compile keywords into a list of arguments
    compile_kwds_list = shlex.split(code_compiler_kwds)

    # Some HeCBench kernels (e.g., entropy) #include a local header that lives only in the
    # kernel's original source folder. LASSI translates only the main source file, never the
    # header, and generated code is written to a different folder. So the original source
    # folder is added to the compiler's include path and the one canonical header is found
    # no matter where the compiled file lives. (A harmless no-op for kernels with no header.)
    if include_dir:
        compile_kwds_list += ['-I', include_dir]

    if lang == "OMP":
        # The desired name of the compiled program
        output_file = source_file.split('.cpp')[0]
        # Construct the compilation command
        compile_command = [code_compiler] + compile_kwds_list + [source_file, '-o', output_file]
    
    elif lang == "CUDA":
        # The desired name of the compiled program
        output_file = source_file.split('.cu')[0]
        # Construct the compilation command
        compile_command = [code_compiler] + compile_kwds_list + [source_file, '-o', output_file]
        
    print("Compile command: " + str(compile_command))
    
    # Execute the compilation command
    result = subprocess.run(compile_command, capture_output=True, text=True)

    # Check if the compilation was successful
    return_result = ""
    if result.returncode == 0:
        print("\n" + color.BOLD + "Compilation of code successful!" + color.END)
    else:
        # If there was an error, print the error message
        return_result = result.stderr
        print("\n" + color.BOLD + color.RED + "***** Compilation failed." + color.END)
        print(" Error message: ")
        print(return_result)
        
    return return_result, output_file


def execute_code(executable_file, exe_input_parameters):

    # Capture current directory to switch back after code execution
    initial_dir = os.getcwd()

    # Defaults, so a missing executable is reported as a failure instead of raising NameError
    return_code = -1
    runtime = None

    # Extract the compiled program to execute
    directory_path = os.path.dirname(executable_file)
    executable = os.path.basename(executable_file)

    # Change the working directory to the subdirectory of the executable
    os.chdir(directory_path)

    # Ensure the binary has execute permissions
    subprocess.run(['chmod', '+x', executable])

    # Build execute command
    execute_command = ['./' + executable] + exe_input_parameters
    print("Execute command: " + str(execute_command) + "\n")

    # Execute the binary object with input parameters, if provided
     
    # Ensure the file is open and available before executing
    if os.path.exists(executable):
        try:
            # Open the file to ensure it is accessible
            with open(executable, 'rb') as code_file:
                # Start the process
                start_runtime = time.time()
                process = subprocess.Popen(execute_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout_lines = []
                stderr_lines = []
                exception_error = ""
                
                # Print the output in real-time
                try:
                    # Read stdout in a separate thread or in a non-blocking way
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        print(line, end='')  # Print each line as it is received
                        stdout_lines.append(line)
                    process.stdout.close()
            
                    # Read stderr in a separate thread or in a non-blocking way
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        stderr_lines.append(line)
                    process.stderr.close()
                
                    # Wait for the process to complete and get the return code
                    return_code = process.wait()
                    runtime = round(time.time() - start_runtime, 4)  # wall-clock seconds
                    
                    if return_code != 0:
                        error_output = ''.join(stderr_lines)
                        print("Program Execution Error:", error_output)

                except Exception as e:
                    print("An error occurred while executing the program: ", e)
                    exception_error = str(e)
        except IOError as e:
            print(f"Pipeline error | An I/O error occurred: {e.strerror}")
        except Exception as e:
            print(f"Pipeline error | An error occurred: {e}")
    else:
        print(f"Pipeline error | The file {executable} does not exist.")
    
    
    # Check if the program ran successfully
    return_result = "0"  # default OK
    if return_code == 0:
        print("\n" + color.BOLD + "Program executed successfully!" + color.END)
    else:
        # If there was an error during execution, print the error message
        return_result = "Program exited with return code: " + str(return_code) + " "
        if return_code == -11:
            return_result += "Segmentation fault detected."
        stderr_output = ''.join(stderr_lines)
        # stdout_output = ''.join(stdout_lines)
        # if stdout_output:
        #    return_result += "\nStandard Output: " + stdout_output
        if exception_error != "":
            return_result += "\nException Error: " + exception_error
        if stderr_output:
            return_result += "\nStandard Error: " + stderr_output
        print("\n" + color.BOLD + color.RED + "***** Execution failed." + color.END)
        print("Error message: ")
        print(return_result)
    
    os.chdir(initial_dir)

    stdout_output = ''.join(stdout_lines)
    
    return return_result, stdout_output, runtime
