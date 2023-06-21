import sys
def trace_func(frame, event, arg):
    # print("event",event,"arg",arg)
    print(frame.f_lineno,frame.f_code)
    if event.lower() == 'exception':
        exc_type, exc_value, exc_traceback = arg
        # handle the exception here
        print(f"Caught exception: {exc_type.__name__}: {exc_value}")
        # remove the exception from the traceback
        return lambda exc_type, exc_value, exc_traceback: None
    return trace_func

if 'with settrace':
    sys.settrace(trace_func)
    raise Exception
    print('cant touch this')
    # Your code here
    sys.settrace(None)
if 'without settrace':
    print('should reach this ')
    raise Exception
    print('cant touch this ')