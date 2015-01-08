# This file is configured by CMake automatically as DartConfiguration.tcl
# If you choose not to use CMake, this file may be hand configured, by
# filling in the required variables.


# Configuration directories and files
SourceDirectory: /users/visics/rbenenso/code/references/cuda/cudatemplates
BuildDirectory: /users/visics/rbenenso/code/references/cuda/cudatemplates

# Site is something like machine.domain, i.e. pragmatic.crd
Site: vesta

# Build name is osname-revision-compiler, i.e. Linux-2.4.2-2smp-c++
BuildName: Linux-g++-4.3

# Submission information
IsCDash: TRUE
DropSite: cdash.icg.tugraz.at
DropLocation: /submit.php?project=cudatemplates
DropSiteUser: 
DropSitePassword: 
DropSiteMode: 
DropMethod: http
TriggerSite: http://cdash.icg.tugraz.at/cgi-bin/Submit-Random-TestingResults.cgi
ScpCommand: /usr/bin/scp

# Dashboard start time
NightlyStartTime: 04:00:00 CET

# Commands for the build/test/submit cycle
ConfigureCommand: "/usr/bin/cmake" "/users/visics/rbenenso/code/references/cuda/cudatemplates"
MakeCommand: /usr/bin/gmake -i

# CVS options
# Default is "-d -P -A"
CVSCommand: /usr/bin/cvs
CVSUpdateOptions: -d -A -P

# Subversion options
SVNCommand: /usr/bin/svn
SVNUpdateOptions: 

# Generic update command
UpdateCommand: /usr/bin/svn
UpdateOptions: 
UpdateType: svn

# Dynamic analisys and coverage
PurifyCommand: 
ValgrindCommand: 
ValgrindCommandOptions: 
MemoryCheckCommand: /usr/bin/valgrind
MemoryCheckCommandOptions: 
MemoryCheckSuppressionFile: 
CoverageCommand: /usr/bin/gcov

# Testing options
# TimeOut is the amount of time in seconds to wait for processes
# to complete during testing.  After TimeOut seconds, the
# process will be summaily terminated.
# Currently set to 25 minutes
TimeOut: 1500
