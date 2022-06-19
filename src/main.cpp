#include <iostream>
#include "vision_protectpravicy.hpp"


int main(int argc, char* argv)
{   
    // 0. get config file
    std::string configfile = argv[1];

    // 1. create instance and initialize
    ProtectPravicy* protector = new ProtectPravicy(configfile);

    // 2. run protector
    protector->removePersonalInfo();

    // 3. release protector instance
    protector->~ProtectPravicy();
    delete protector;

    return 0;
}