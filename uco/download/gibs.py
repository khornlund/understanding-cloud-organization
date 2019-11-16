"""
Downloads data from:
https://wiki.earthdata.nasa.gov/display/GIBS/GIBS+API+for+Developers
"""
from pathlib import Path
from dateutil.rrule import rrule, DAILY
from datetime import date
from calendar import monthrange
from concurrent.futures import as_completed, ThreadPoolExecutor

from owslib.wms import WebMapService
from tqdm import tqdm

from uco.utils import setup_logger


class DomainBase:
    @property
    @classmethod
    def name(cls):
        raise NotImplementedError

    @property
    @classmethod
    def bbox(cls):
        raise NotImplementedError

    @property
    @classmethod
    def valid_months(cls):
        raise NotImplementedError


class DomainA:
    """
     -61◦E -40◦E; 10◦N 24◦N DJF, MAM
    """

    name = "domain_a"
    bbox = (-61, 10, -40, 24)
    valid_months = [1, 2, 3, 4, 5, 12]


class DomainB:
    """
    159◦E 180◦E; 8◦N 22◦N DJF
    """

    name = "domain_b"
    bbox = (159, 8, 180, 22)
    valid_months = [1, 2, 12]


class DomainC:
    """
    -135◦E -114◦E; -1◦N -15◦N DJF, SON
    """

    name = "domain_c"
    bbox = (-135, -15, -114, -1)
    valid_months = [1, 2, 9, 10, 11, 12]


class TimeGenerator:
    """
    Generate dates in Y-m-d format for the valid months of a domain.
    """

    valid_years = [2002, 2003, 2004, 2005, 2018]

    @classmethod
    def for_domain(cls, domain):
        for year in cls.valid_years:
            for month in domain.valid_months:
                start_date = date(year, month, 1)
                ndays = monthrange(year, month)[1]
                for dt in rrule(DAILY, dtstart=start_date, count=ndays):
                    yield dt.strftime("%Y-%m-%d")


class MapDownloader:

    url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"

    H = 1400
    W = 2100

    SRS = "EPSG:4326"
    format = "image/jpeg"

    layers = [
        "MODIS_Terra_CorrectedReflectance_TrueColor",
        "MODIS_Aqua_CorrectedReflectance_TrueColor",
    ]

    def __init__(self, data_dir, nworkers=4, verbose=2):
        self.data_dir = Path(data_dir)
        self.nworkers = nworkers
        self.logger = setup_logger(self, verbose)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self):
        requests = self.setup_requests()
        self.logger.info(
            f"Processing {len(requests)} downloads with {self.nworkers} workers to "
            f'"{self.data_dir}"'
        )
        with ThreadPoolExecutor(max_workers=self.nworkers) as executor:
            results = [executor.submit(request_wms, **request) for request in requests]
            for future in tqdm(as_completed(results), total=len(requests)):
                pass

    def setup_requests(self):
        wms = self.connect_wms()
        requests = []
        for domain in [DomainA, DomainB, DomainC]:
            for dt in TimeGenerator.for_domain(domain):
                for layer in self.layers:
                    filename = f"{dt}-{domain.name}-{layer}.jpg".replace("_", "-")
                    filename = self.data_dir / filename
                    requests.append(
                        {
                            "wms": wms,
                            "layer": layer,
                            "srs": self.SRS,
                            "bbox": domain.bbox,
                            "size": (self.W, self.H),
                            "format": self.format,
                            "time": dt,
                            "save_as": filename,
                        }
                    )
        self.logger.info(f"Setup {len(requests)} requests")
        return requests

    def connect_wms(self):
        self.logger.info(f'Connecting to WMS: "{self.url}"')
        wms = WebMapService(self.url)
        return wms


def request_wms(wms, layer, srs, bbox, size, format, time, save_as):
    try:
        r = wms.getmap(
            layers=[layer], srs=srs, bbox=bbox, size=size, format=format, time=time
        )
        with open(save_as, "wb") as fh:
            fh.write(r.read())
    except Exception as ex:
        print(f"Caught exception for {save_as.name}: {ex}")
