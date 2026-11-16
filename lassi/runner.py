"""Compile and execute code, capturing output and runtime."""
import os
import shlex
import subprocess
import threading
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


def execute_code(executable_file, exe_input_parameters, echo=True, timeout=None):
    """
    Run a compiled program from its own folder and capture what it prints.

    Returns (result, stdout, runtime_seconds). `result` is "0" on success, otherwise a message
    describing the failure. `runtime_seconds` is the wall-clock time of the program itself, from
    process start to process exit (compile time and the time spent relaying output are not
    included); it is None if the program could not be started.

    echo:    print the program's output as it arrives (False for quiet repeat runs)
    timeout: kill the program after this many seconds and report a failure (None = no limit)
    """
    # Capture current directory to switch back after code execution
    initial_dir = os.getcwd()

    return_code = -1
    runtime = None
    stdout_lines = []
    stderr_lines = []
    exception_error = ""

    # Extract the compiled program to execute
    directory_path = os.path.dirname(executable_file)
    executable = os.path.basename(executable_file)

    # Change the working directory to the subdirectory of the executable
    os.chdir(directory_path)

    # Ensure the binary has execute permissions
    subprocess.run(['chmod', '+x', executable])

    # Build execute command
    execute_command = ['./' + executable] + exe_input_parameters
    if echo:
        print("Execute command: " + str(execute_command) + "\n")

    # Ensure the file is available before executing
    if os.path.exists(executable):
        try:
            start_runtime = time.perf_counter()
            process = subprocess.Popen(execute_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Drain both streams at the same time so neither pipe can fill up and stall the program.
            def drain(stream, lines, show):
                for line in stream:
                    if show:
                        print(line, end='')  # Print each line as it is received
                    lines.append(line)
                stream.close()

            readers = [threading.Thread(target=drain, args=(process.stdout, stdout_lines, echo), daemon=True),
                       threading.Thread(target=drain, args=(process.stderr, stderr_lines, False), daemon=True)]
            for reader in readers:
                reader.start()

            try:
                return_code = process.wait(timeout=timeout)
                runtime = round(time.perf_counter() - start_runtime, 4)  # wall-clock seconds, process only
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                exception_error = f"Timed out after {timeout} seconds and was stopped."
            for reader in readers:
                reader.join()

            if return_code != 0 and echo:
                print("Program Execution Error:", ''.join(stderr_lines))
        except Exception as e:
            print(f"Pipeline error | An error occurred: {e}")
            exception_error = str(e)
    else:
        print(f"Pipeline error | The file {executable} does not exist.")

    # Check if the program ran successfully
    return_result = "0"  # default OK
    if return_code == 0:
        if echo:
            print("\n" + color.BOLD + "Program executed successfully!" + color.END)
    else:
        # If there was an error during execution, print the error message
        return_result = "Program exited with return code: " + str(return_code) + " "
        if return_code == -11:
            return_result += "Segmentation fault detected."
        stderr_output = ''.join(stderr_lines)
        if exception_error != "":
            return_result += "\nException Error: " + exception_error
        if stderr_output:
            return_result += "\nStandard Error: " + stderr_output
        if echo:
            print("\n" + color.BOLD + color.RED + "***** Execution failed." + color.END)
            print("Error message: ")
            print(return_result)

    os.chdir(initial_dir)

    stdout_output = ''.join(stdout_lines)

    if echo and runtime is not None:
        print("\nProgram runtime (s): " + str(runtime) + "\n")

    return return_result, stdout_output, runtime
