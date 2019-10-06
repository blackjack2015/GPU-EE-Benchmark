OBJ = o

ARCH = 70

NAME = fp64_fma

A2-FP64: $(NAME).$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda

$(NAME).$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -Xptxas -O0,-dlcm=cv,-dscm=wt  -c $(INCLUDES) $<

clean:
	rm -f $(NAME).$(OBJ) $(NAME) A2-FP64
