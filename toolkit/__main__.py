import argparse
import crowdscan
import os
from crowdscan.tests import testcrowd, testload, testmetrics

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CrowdScan - Human Trajectory Analysis Package')
    
    parser.add_argument('--run-tests', '--r',
                           default='none', choices=['none', 'all', 'testload', 'testcrowd','testmetrics'],
                           help='pick a test to run and check the code'
                                '(default: "none")')
    parser.add_argument('--separator', '--sep',
                           default='" "', choices=['" "', ',', ';'],
                           help='select the default separator for input and output files'
                                '(default: " ")')

    args = parser.parse_args()

    # Running tests

    if not args.run_tests == 'none':
        if args.run_tests == 'testcrowd' or args.run_tests == 'all':
            testcrowd.run()

        if args.run_tests == 'testload' or args.run_tests == 'all':
            module_directory = os.path.dirname(crowdscan.__file__)
            testload.run(module_directory, args)

        if args.run_tests == 'testmetrics' or args.run_tests == 'all':
            module_directory = os.path.dirname(crowdscan.__file__)
            testmetrics.run(module_directory, args)

    # End of program

    print("\n###############################\n# CrowdScan : jobs done. Bye! #\n###############################\n")

