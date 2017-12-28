#check all files using pyflakes.
#this is in a makefile because then I can use vim to quickly squash all these errors.
check:
	pyflakes *.py
	pyflakes src/*.py
