#ifndef _TIMER_H_
#define _TIMER_H_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>


class Timer{

  public:
    Timer(const std::string& timer_tag);
    ~Timer();
     void check();
     void check(const std::string& statement);
     void mute();
     void stop();    

  private:
    std::string tag;
    double init_time;
    double last_time;
    double this_time;
    bool _mute;

};


#endif  // _TIMER_H_
