OBJ = o

ARCH = 70

NAME = special_function

all: $(NAME) A3-SF

A3-SF: $(NAME)_fast_math.$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda --use_fast_math

$(NAME)_fast_math.$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -c $(INCLUDES) -o $(NAME)_fast_math.$(OBJ) $< --use_fast_math

$(NAME): $(NAME).$(OBJ)
	nvcc -arch=sm_$(ARCH) -o $@ $< -lcuda

$(NAME).$(OBJ): $(NAME).cu
	nvcc -arch=sm_$(ARCH) -c $(INCLUDES) $<

clean:
	rm -f $(NAME).$(OBJ) $(NAME) $(NAME)_fast_math $(NAME)_fast_math.$(OBJ) A3-SF
