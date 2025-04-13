import logging


def setup_logger(disable_logging: bool = False, log_level: int = logging.INFO, log_file: str = None) -> None:
    """
    Настройка логгирования.

    :param disable_logging: Если True, логгирование отключается.
    :param log_level: Уровень логгирования (по умолчанию INFO).
    :param log_file: Имя файла для записи логов (если None, логи выводятся в консоль).
    """
    if disable_logging:
        logging.disable(logging.CRITICAL)
        return

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=log_level, 
        format=log_format,
        filename=log_file,
        filemode='a'
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("Настройка логгера завершена.")
