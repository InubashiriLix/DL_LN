import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", '--output',

                    # action='store_true',
                    # the action means weather the flag is
                    # true or false (not require the value in command)


                    required=True,
                    # the required option means the option is required or not
                    # it does not stand for the value is need or not !!!

                    help="shows output"
                    )
args = parser.parse_args()

if args.output:
    print("This is some output")
    print(args.output)



