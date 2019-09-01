.PHONY: tester
tester:
	g++ -O4 -pthread -std=c++11 -o tester tester.cpp kdt.cpp

.PHONY: clean
clean:
	rm -f out *.dat *.o tester kdt core vgcore.* *~ *.h.gch

