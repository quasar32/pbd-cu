.PHONY: clean

pbd: pbd.o
	nvcc $^ -o $@ -lm 

pbd.o: pbd.cu
	nvcc $< -o $@ -c 

vid: vid.o glad/src/gl.o
	gcc $^ -o $@ -lSDL2main -lSDL2 -lm -lswscale \
		-lavcodec -lavformat -lavutil -lx264 
	
vid.o: vid.c
	gcc $< -o $@ -c -Iglad/include

glad/src/gl.o: glad/src/gl.c
	gcc $< -o $@ -c -Iglad/include 

clean:
	rm glad/src/gl.o *.o pbd vid *.mp4 *.csv -f
