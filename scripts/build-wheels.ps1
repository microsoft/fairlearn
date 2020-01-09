# Script to build wheels and drop the version file
param(
    [Parameter(Mandatory)]
    [string]$targetType,
    [Parameter(Mandatory)]
    [uint64]$devVersion,
    [Parameter(Mandatory)]
    [string]$versionFilename
)

$allowedTargetTypes = @("Test", "Prod")

if( !$allowedTargetTypes.Contains($targetType) )
{
    throw "Unrecognised targetType: $targetType"
}

if( Test-Path env:FAIRLEARN_DEV_VERSION )
{
    throw "Environment variable FAIRLEARN_DEV_VERSION must not be set"
}

# Install ourselves
pip install -e python/fairlearn-core/.
pip install -e python/fairlearn/.

# Update the ReadMe file
# Do this before setting FAIRLEARN_DEV_VERSION so that the links
# on Test-PyPI are still correct
Write-Host
Write-Host "Updating ReadMe file"
$readMeScript = Join-Path -resolve scripts process_readme.py
$target = Join-Path -resolve $(Get-Location) README.md
python $readMeScript --input $target --output $target --loglevel INFO
if ($LASTEXITCODE -ne 0)
{
    throw "process_readme.py failed with result code $LASTEXITCODE"
}
Write-Host
Write-Host "Copying REAMDE.md to package directories python/fairlearn-core and python/fairlearn"
cp README.md python/fairlearn-core/.
cp README.md python/fairlearn/.


# Set the environment variable for test if required
if( $targetType -eq "Test" )
{
    $Env:FAIRLEARN_DEV_VERSION = $devVersion
}

# Store fairlearn version (including FAIRLEARN_DEV_VERSION) in the specified file
Write-Host "Storing fairlearn version in $versionFilename"
$versionScript = Join-Path -resolve scripts fairlearn_version.py
python $versionScript > $versionFilename
if ($LASTEXITCODE -ne 0)
{
    throw "fairlearn_version.py failed with result code $LASTEXITCODE"
}

# Create the packages
Write-Host
Write-Host "Creating Packages"
Write-Host (Get-Item -Path ".\").FullName
python python/fairlearn-core/setup.py sdist bdist_wheel
python python/fairlearn/setup.py sdist bdist_wheel

Write-Host
Write-Host "Package created"