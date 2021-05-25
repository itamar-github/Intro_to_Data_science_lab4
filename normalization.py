from point import Point
from numpy import mean, var


class DummyNormalizer:
    def fit(self, points):
        pass

    def transform(self, points):
        return points


class ZNormalizer:
    def __init__(self):
        self.mean_variance_list = []

    def fit(self, points):
        all_coordinates = [p.coordinates for p in points]
        self.mean_variance_list = []
        for i in range(len(all_coordinates[0])):
            values = [x[i] for x in all_coordinates]
            self.mean_variance_list.append([mean(values), var(values, ddof=1)**0.5])

    def transform(self, points):
        new = []
        for p in points:
            new_coordinates = p.coordinates
            new_coordinates = [(new_coordinates[i] - self.mean_variance_list[i][0]) / self.mean_variance_list[i][1]
                               for i in range(len(p.coordinates))]
            new.append(Point(p.name, new_coordinates, p.label))
        return new


class SumNormalizer:
    def __init__(self):
        self.sum_of_values_list = []

    def fit(self, points):
        """
        fit the sum_of_values list to the given set of points
        :param points: List of Points
        :return: None
        """
        all_coordinates = [point.coordinates for point in points]
        self.sum_of_values_list = []
        # loop over the coordinates
        for i in range(len(all_coordinates[0])):
            self.sum_of_values_list.append(sum([abs(coordinates[i]) for coordinates in all_coordinates]))

    def transform(self, points):
        """
        transform the list of points according to L1 norm
        :param points: List of Points
        :return: List of normalized Points
        """
        new = []
        n_coordinates = len(points[0].coordinates)
        for point in points:
            new_coordinates = [(point.coordinates[i])/(self.sum_of_values_list[i]) for i in range(n_coordinates)]
            new.append(Point(point.name, new_coordinates, point.label))

        return new


class MinMaxNormalizer:
    def __init__(self):
        self.min_list = []
        self.max_list = []

    def fit(self, points):
        """
        fit the min_max_list to the given set of points
        :param points: List of Points
        :return: None
        """
        all_coordinates = [point.coordinates for point in points]
        self.min_list = []
        self.max_list = []
        # loop over the coordinates
        n_coordinates = len(points[0].coordinates)
        for i in range(n_coordinates):
            self.min_list.append(min([coordinates[i] for coordinates in all_coordinates]))
            self.max_list.append(max([coordinates[i] for coordinates in all_coordinates]))

    def transform(self, points):
        """
        transform the list of points according to L1 norm
        :param points: List of Points
        :return: List of normalized Points
        """
        new = []
        n_coordinates = len(points[0].coordinates)
        min_max_range = []

        for i in range(n_coordinates):
            min_max_range.append(self.max_list[i] - self.min_list[i])

        for point in points:
            new_coordinates = [(point.coordinates[i] - self.min_list[i])/min_max_range[i] for i in range(n_coordinates)]
            new.append(Point(point.name, new_coordinates, point.label))

        return new
