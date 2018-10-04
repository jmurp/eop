#include <iostream>
#include <sstream>
#include <string>
#include <ctime>

std::string zeropad_number(int n)
{
	std::stringstream ss;
	ss << n;
	std::string str;
	ss >> str;

	int len = str.length();
	for (int i = 0; i < 4 - len; i++) {
		str = "0" + str;
	}
	return str;
};


int main(){

	std::cout << zeropad_number(1) << std::endl;
	std::cout << zeropad_number(28) << std::endl;

	return 0;
};
