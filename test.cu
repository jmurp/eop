#include <iostream>
#include <string>
#include <ctime>


int main(){

	char stamp[20];
	time_t t = time(0);
	strftime(stamp, sizeof(stamp), "%Y-%m-%d-%X", gmtime(&t));
	std::cout << "stamp is: " << stamp << std::endl;

	return 0;
};
