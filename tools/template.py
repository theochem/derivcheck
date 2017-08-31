#!/usr/bin/env python

template_dict = {
    "TPL_ANACONDA_TOKEN": "h6KCoaPtcyTfCeZeCk3cp/avreJai6B/qh5674TO0OesClzpzKjxqfAUThKU2L+Zkjc9qET9n6YXa9iwmaTcZ9xcqr0mDIXMDttsqXU57nOR9mLraZrSARbWKJltLVbC3fYyHlZhs1Z502chgIuMSkIJx1lb5gj4by7cv2tL4BEiyQ0C8XfbxTvw8xiJVS3nmOgMtDe+2PJUmyVpQAXFJHOJhw5aCHLNv9ci478DrijkWKp4QjJ9QHa3vnT3fwccvrQwGbNqM4yZwSGOIKyqUSBVLGbd+fQQgyT3WtF1vvmqRoj48+8E3JL4iTlmPGLlFvIm9e0PI1w9CY+wSlJEN5TEy4U6qrbkiOARhqtK5wwp4MGjDVsLY6FjADStzKtI84eKIHbWkm/AodjjZYC1kFBpzlbq/YMhXEG0EkbNgqqD3yQ1NyCRM+1SnRaaiZ4CwdRsf73yTxxf8F99aJxpxRt7oz2Uhdvit8LKxPG2iGOsLSMYQU75pF1Ipvvz3rDYXHNsAKrYpCm5D+C4LXWutdS9w17kUgYKZ0AbXpSaxA5GC79Oh7obx7Yp9OGjANpJBT8wSH6/xvufSwZhS2BXzu6quvgbndDDb9chPsxOgKW2aePomPtouN7UPR9yAYh5tNCn5rxNKQ3qXWFxw/1OuYGbBv/xxDiE0cXaLuDNsTM=",
    "TPL_GITHUB_TOKEN": "tBZxpaEmIsNl9/5Vb3BlOP5Q6k5aKvYLWhbOdvSw8uITW9nbrsySNPWvf5uzHqrTVTmnL7fS5Sout3TvJWUVmRknrW/JXrlM/LEEZooLK0WzoL48Hk/iQziO0fVBvsLs4RG4OQ4yGjQbjOwU5A/QnA0tzY5NXoJ74EgzF9VORRnRFiJiYsy5xN703CfkiiH/WFJOM9jZxg60JWRdPWsaYC5W/wMFqQTLw8BGZujqE8gwzL7a7h4QaCspHfJQQDNbYoulx1lP+dw5BXwMffTkMRr1ZJEBHwVbvF7zaRypomlTPab3ygH9m3vYQuXHAD85GjFsi5DQL7BSzMUF/dnjVe9gWJ0MVSTGRpkUEZQ+fLv0kBo/hth7hWeF+P0BLhnbiESj+K23JhqtruBCOJ9g81Tss/j/Sgloip16B2bUopUQa5VD7Sp1uLtpfSdBjBb3epJI0Pz/Nx4J+mlsWsQAKwO/gqYdrSnDkuqssF5jZVNgFYSqLjMBWSvLAjAbzCYBvSxIqd4eKi69OoLMOZlBfLepArTwmBflmQhs33nFCMYUW0ZhzYLHMnP70phFvZaaEVwnZew4qhj6wXqcCyvLS8U2I06BTJfu9O0Cwcz49SxVIugIUms3fdeBqQnPqXse4VlEAdgZc2wmtBkdTuZk9quV4IDr8JAtZFUXMAGBL30=",
    "TPL_PYPI_PASSWORD": "PLRDqMVqDu6t5i1sixypiIvmlRyPMnsS1G6CnPC8gYbVRQ4mqpNUdX17UxinQlnxB0crzeSzEwFQ7AUxJDNoGYr0q09olfD7w68yg/qsGfBd5Zke3P1oa19WzaizBAIgsC73wMb0tOa9HhoLV/5WlIENNtChtXtgv1z9Oc9upnpOietvUXD1zbsa82xK5QzNFqQF9aPu7Q3OaZ3e9NtPwoQVPMuRd0O3kj1vFqG2Kb0zbLfUjrrmuZgJzSIif9G8OmnAfRjNBqVItFuMsnA9WnolrPcx7tzySbMR+Mrax/tehO6MLb+P8kriSg9B1vBr5B2r8fDN14HWaQwsSyg58VmpQjdgjo1UJXPzB3jEYLY/lF4BqakIIL4se+Er13AVKWrJTI/IEMr1ztKlFKqMJPk4aLhG9DVLSOgebG7ZcH8ASKSnA6a2zVxog9FgaYZnZkIlO+lUfQpd338piqeK+wz5QaILIt3UVBln6nMB5GoqG8gyV7xUOjJyyRF5tx7tafRXGdCb6sRANiv5PKhYZBLvIscH7ITJt73EZBCrWRI2mqPgx+cjzOlA6OFo1eslGBR2ACLfE/+qAh0koP8JibqK/WWLen0cTAmdIVa2x3rNK1EepEi7Tow4G3fjwzrFvLzLc8DyFXt1Tt9swE6/QjNOb9Y8gC65PLjviDrNw8o=",
    "PROJECT_NAME": "derivcheck",
    "GITHUB_REPO_NAME": "theochem/derivcheck",
}

import sys
from string import Template

template_fn = sys.argv[1]

with open(template_fn) as fh:
    l = fh.read()

t = Template(l)
s = t.safe_substitute(**template_dict)
with open(".travis.yml", "w") as fh:
    fh.write(
        "#\n# THIS FILE IS AUTOGENERATED. DO NOT EDIT DIRECTLY. CHANGES IN \".travis.yml.tpl\" INSTEAD.\n#\n")
    fh.write(s)
