#include <iostream>
#include "inferer.h"

#include <fstream>
#include <string>
#include "strtokenizer.h"

using namespace std;

int main()
{

	Inferer inf;
	inf.init_inference();
	inf.infer();

	return 0;
}