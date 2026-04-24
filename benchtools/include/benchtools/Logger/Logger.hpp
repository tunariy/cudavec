#pragma once

#include <spdlog/spdlog.h>

#include <memory>

class Logger {
public:
  Logger();

  void Init();

  std::shared_ptr<spdlog::logger>& GetLogger() { return m_Logger; }

private:
  std::shared_ptr<spdlog::logger> m_Logger;
};

inline Logger g_Logger;

#define TRACE(...) ::g_Logger.GetLogger()->trace(__VA_ARGS__)
#define INFO(...) ::g_Logger.GetLogger()->info(__VA_ARGS__)
#define WARN(...) ::g_Logger.GetLogger()->warn(__VA_ARGS__)
#define ERR(...) ::g_Logger.GetLogger()->error(__VA_ARGS__)
#define CRITICAL(...) ::g_Logger.GetLogger()->critical(__VA_ARGS__)