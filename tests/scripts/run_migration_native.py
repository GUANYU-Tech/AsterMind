"""Real copy/move native regression under both global compute modes."""
import sys
from run_compute_comm import main
if __name__=='__main__':
    sys.argv[1:1]=['--cases','DS06-copy','DS06-move']
    main()
