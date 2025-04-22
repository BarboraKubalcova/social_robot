#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <JHPWMPCA9685.h>

// rozsah sa potom zmeni podla 3D modelu
int servoMin = 20 ;
int servoMax = 180 ;



int map ( int x, int in_min, int in_max) {
    int toReturn =  (x - in_min) * (servoMax - servoMin) / (in_max - in_min) + servoMin ;
    // For debugging:
    // printf("MAPPED %d to: %d\n", x, toReturn);
    return toReturn ;
}

void set_angle(int angle, int channel){
    // pca9685->setPWM(channel,0, map(angle, 0, 180));

    PCA9685 *pca9685 = new PCA9685() ;
    int err = pca9685->openPCA9685();

    if (err < 0){
        printf("Error: %d", pca9685->error);
    } else {
        printf("PCA9685 Device Address: 0x%02X\n",pca9685->kI2CAddress) ;
        pca9685->setAllPWM(0,0) ;
        pca9685->reset() ;
        pca9685->setPWMFrequency(60) ;
        pca9685->setPWM(0,0, map(angle, 0, 180));
        sleep(1);
    }
    pca9685->closePCA9685();
}


int main() {
    set_angle(90, 0);
    
}
