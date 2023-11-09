param (
    [switch]$rebuild = $false,
    [string]$testdata = "C:\Users\weber\Nextcloud3\ER-C-Data\adhoc\libertem\libertem-test-data\",
    [string]$cuda = ""
)

if ($rebuild) {
    $rebuildflag = "-r"
} else {
    $rebuildflag = ""
}

if ($cuda) {
    $cudaflag = "-cuda$cuda"
} else {
    $cudaflag = ""
}

#tox -r -e py39 -- -m "not dist" tests | tee tests.log
tox $rebuildflag -e py37$cudaflag,py38$cudaflag,py39$cudaflag,py310$cudaflag,py311$cudaflag -- -m "not dist" tests | tee tests.log
$Env:TESTDATA_BASE_PATH = $testdata
#tox -r -e py39-data -- -m "not dist" | tee data-tests.log
tox $rebuildflag -e py37-data$cudaflag,py38-data$cudaflag,py39-data$cudaflag,py310-data$cudaflag,py311-data$cudaflag -- -m "not dist" | tee data-tests.log
tox $rebuildflag -e notebooks$cudaflag | tee notebook-tests.log
