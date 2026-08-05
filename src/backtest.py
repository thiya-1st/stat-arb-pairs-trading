def calculate_position_size(capital, beta, price_y, price_x):
    dollars_in_y = capital / (1 + beta)
    dollars_in_x = capital - dollars_in_y
    shares_y = dollars_in_y / price_y
    shares_x = dollars_in_x / price_x
    return shares_x, shares_y

